"""
Locality Sensitive Hashing (LSH)

It creates random hyperplanes to hash vectors into a vector of 0s and 1s of whether the vector is on the positive or negative side of the hyperplane.
At search time, it only searches the `nprobe` areas that are close to the query vector.

k = number of hyperplanes in the hash table
l = number of hash tables
n = number of vectors
d = dimension of the vectors

Build time complexity: O(n*d)
Search time complexity: O(n^p) for 0 < p < 1.
Memory usage: O(n*d)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional
from collections import defaultdict
from stackdb.indexes.base_index import BaseIndex
from stackdb.models.chunk import Chunk
from stackdb.filter import ChunkFilter


class LSHIndex(BaseIndex):
    def __init__(
        self, dimension: int, num_tables: int = 10, hash_size: int = 10, seed: int = 42
    ):
        """
        num_tables: Number of hash tables.
        hash_size: Size of the hash.
        """
        super().__init__(dimension)
        self.num_tables = num_tables
        self.hash_size = hash_size

        self.hash_tables: List[Dict[str, Set[str]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        self.random_vectors: List[np.ndarray] = []
        self.seed = seed
        self._generate_random_vectors(seed)

    def _generate_random_vectors(self, seed: int) -> None:
        np.random.seed(seed)

        for _ in range(self.num_tables):
            random_matrix = np.random.randn(self.hash_size, self.dimension)
            random_matrix = random_matrix / np.linalg.norm(
                random_matrix, axis=1, keepdims=True
            )

            self.random_vectors.append(random_matrix)

    def _hash_vectors(self, vectors: np.ndarray, table_idx: int) -> str:
        projections = vectors @ self.random_vectors[table_idx].T
        binary_vectors = (projections > 0).astype(int)
        return ["".join(binary_vector.astype(str)) for binary_vector in binary_vectors]

    def add_vectors(self, chunks: List[Chunk]) -> None:
        self._verify_add_chunks(chunks)

        new_vectors = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        self.chunks.update({chunk.id: chunk for chunk in chunks})

        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.concatenate([self.vectors, new_vectors], axis=0)

        new_chunk_ids = [chunk.id for chunk in chunks]
        self.chunk_ids.extend(new_chunk_ids)

        for table_idx in range(self.num_tables):
            hash_values = self._hash_vectors(new_vectors, table_idx)
            for hash_value, chunk_id in zip(hash_values, new_chunk_ids):
                self.hash_tables[table_idx][hash_value].add(chunk_id)

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

        for table_idx in range(self.num_tables):
            hash_values = self._hash_vectors(self.vectors[indices_to_delete], table_idx)
            for hash_value, chunk_id in zip(
                hash_values, [self.chunk_ids[idx] for idx in indices_to_delete]
            ):
                self.hash_tables[table_idx][hash_value].discard(chunk_id)
                if not self.hash_tables[table_idx][hash_value]:
                    del self.hash_tables[table_idx][hash_value]

        return True

    def search(
        self,
        query: List[float],
        k: int = 10,
        filter: Optional[str] = None,
        max_candidates: int = 100,
    ) -> List[Tuple[Chunk, float]]:
        """
        max_candidates: Maximum number of candidates to consider (default: 10*k)
        """
        if len(query) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query)} doesn't match index dimension {self.dimension}"
            )

        if self.vectors.shape[0] == 0:
            return []

        filterer = ChunkFilter(filter) if filter else None

        max_candidates = max_candidates or 10 * k

        query_array = np.array([query], dtype=np.float32)
        chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        candidate_ids = set()

        for table_idx in range(self.num_tables):
            hash_value = self._hash_vectors(query_array, table_idx)[0]
            if hash_value in self.hash_tables[table_idx]:
                candidate_ids.update(self.hash_tables[table_idx][hash_value])

        if len(candidate_ids) < max_candidates:
            for table_idx in range(self.num_tables):
                hash_value = self._hash_vectors(query_array, table_idx)[0]

                for bit_pos in range(len(hash_value)):
                    flipped_hash = list(hash_value)
                    flipped_hash[bit_pos] = "1" if flipped_hash[bit_pos] == "0" else "0"
                    flipped_hash_value = "".join(flipped_hash)

                    if flipped_hash_value in self.hash_tables[table_idx]:
                        candidate_ids.update(
                            self.hash_tables[table_idx][flipped_hash_value]
                        )

                if len(candidate_ids) >= max_candidates:
                    break

        candidate_chunk_ids = list(candidate_ids)[:max_candidates]

        if not candidate_chunk_ids:
            return []

        candidate_vector_indices = [chunk_id_to_idx[cid] for cid in candidate_chunk_ids]
        candidate_vectors = self.vectors[candidate_vector_indices]
        distances = (1 - candidate_vectors @ query_array.T).flatten()

        candidate_distances = [
            (self.chunks[cid], distance)
            for cid, distance in zip(candidate_chunk_ids, distances)
            if not filterer or filterer.matches(self.chunks[cid])
        ]
        candidate_distances.sort(key=lambda x: x[1])
        return candidate_distances[:k]

    def build_index(self) -> None:
        pass  # built incrementally

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "lsh",
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
            "seed": self.seed,
        }
