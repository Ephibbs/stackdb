"""
Inverted File (IVF) Index uses k-means clustering to partition vectors into clusters.

At search time, it only searches the `nprobe` nearest clusters.

k = number of clusters
d = dimension of the vectors
i = number of iterations for k-means clustering

Build time complexity: O(n*d*(i+k))
Search time complexity: O(n^p) for 0 < p < 1.
Memory usage: O(n*d)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import KMeans
from collections import defaultdict
from stackdb.indexes.base_index import BaseIndex
from stackdb.models.chunk import Chunk
from stackdb.filter import ChunkFilter


class IVFIndex(BaseIndex):
    def __init__(
        self, dimension: int, num_clusters: int = 100, nprobe: int = 1, seed: int = 42
    ):
        """
        Args:
            dimension: Dimensionality of the vectors
            num_clusters: Number of clusters to create
            nprobe: Number of nearest clusters to search during query
            seed: Random seed for k-means clustering
        """
        super().__init__(dimension)
        self.num_clusters = num_clusters
        self.nprobe = nprobe

        self.centroids: Optional[np.ndarray] = None

        self.inverted_file: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

        self.cluster_assignments: List[int] = []

        self.kmeans = None
        self.is_trained = False
        self.seed = seed

    def _train_clusters(self) -> None:
        if self.vectors.shape[0] == 0:
            return

        n_clusters = min(self.num_clusters, self.vectors.shape[0])

        # K-means clustering using euclidean distance. Approximate to cosine distance.
        # TODO: Implement a cosine distance K-Means.
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        cluster_labels = self.kmeans.fit_predict(self.vectors)

        self.centroids = self.kmeans.cluster_centers_
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.centroids = self.centroids / norms

        self.inverted_file.clear()
        self.cluster_assignments = cluster_labels.tolist()

        for vector_idx, (chunk_id, cluster_id) in enumerate(
            zip(self.chunk_ids, cluster_labels)
        ):
            self.inverted_file[cluster_id].append((chunk_id, vector_idx))

        self.is_trained = True

    def add_vectors(self, chunks: List[Chunk]) -> None:
        self._verify_add_chunks(chunks)

        vectors = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        self.chunks.update({chunk.id: chunk for chunk in chunks})

        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.concatenate([self.vectors, vectors], axis=0)

        self.chunk_ids.extend([chunk.id for chunk in chunks])

        if self.is_trained and self.centroids is not None:
            similarities = vectors @ self.centroids.T
            cluster_assignments = np.argmax(similarities, axis=1)
            self.cluster_assignments.extend(cluster_assignments.tolist())

            for vector_idx, cluster_id in enumerate(cluster_assignments):
                self.inverted_file[cluster_id].append(
                    (chunks[vector_idx].id, vector_idx)
                )
        else:
            self.is_trained = False

    def remove_vectors(self, chunk_ids: List[str]) -> bool:
        existing_chunk_ids = set([cid for cid in chunk_ids if cid in self.chunks])

        if len(existing_chunk_ids) == 0:
            return False

        id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        indices_to_delete = [id_to_idx[cid] for cid in existing_chunk_ids]

        self.vectors = np.delete(self.vectors, indices_to_delete, axis=0)

        self.chunk_ids = [cid for cid in self.chunk_ids if cid not in chunk_ids]
        self.chunks = {
            cid: chunk for cid, chunk in self.chunks.items() if cid not in chunk_ids
        }

        if self.is_trained:
            for cluster_id in self.inverted_file:
                self.inverted_file[cluster_id] = [
                    (cid, vidx)
                    for cid, vidx in self.inverted_file[cluster_id]
                    if cid not in chunk_ids
                ]

            for idx in indices_to_delete:
                cluster_id = self.cluster_assignments[idx]
                self.inverted_file[cluster_id] = [
                    (cid, vidx)
                    for cid, vidx in self.inverted_file[cluster_id]
                    if cid not in chunk_ids
                ]

            self.cluster_assignments = [
                cluster_id
                for cluster_id in self.cluster_assignments
                if self.chunks[self.chunk_ids[cluster_id]] not in chunk_ids
            ]

        return True

    def search(
        self,
        query: List[float],
        k: int = 10,
        filter: Optional[str] = None,
        nprobe: int = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        nprobe: Number of clusters to search < k.
        """
        if len(query) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query)} doesn't match index dimension {self.dimension}"
            )

        if not self.is_trained:
            self.build_index()

        filterer = ChunkFilter(filter) if filter else None

        if nprobe is None:
            nprobe = self.nprobe

        nprobe = min(nprobe, len(self.centroids))

        query_array = np.array([query], dtype=np.float32)

        similarities = query_array @ self.centroids.T
        nearest_clusters = np.argsort(similarities, axis=1)[:, :nprobe].flatten()

        candidate_ids = []
        for cluster_id in nearest_clusters:
            if cluster_id in self.inverted_file:
                candidate_ids.extend(self.inverted_file[cluster_id])

        if not candidate_ids:
            return []

        candidate_vectors = self.vectors[[vidx for _, vidx in candidate_ids]]
        distances = (1 - query_array @ candidate_vectors.T).flatten()

        filtered_candidates = []
        for chunk_id, distance in zip([cid for cid, _ in candidate_ids], distances):
            chunk = self.chunks[chunk_id]
            if not filterer or filterer.matches(chunk):
                filtered_candidates.append((chunk, distance))
        filtered_candidates.sort(key=lambda x: x[1])
        return filtered_candidates[:k]

    def build_index(self) -> None:
        """
        This trains the k-means clustering and builds the inverted file
        """
        if self.vectors.shape[0] == 0:
            return

        self._train_clusters()

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "ivf",
            "num_clusters": self.num_clusters,
            "nprobe": self.nprobe,
        }
