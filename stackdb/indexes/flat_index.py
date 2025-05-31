"""
Flat Index Implementation

It's a brute-force approach that calculates distances between the query vector and all other vectors in the index.

n = number of vectors
d = dimension of the vectors

Build time complexity: O(n*d)
Search time complexity: O(n*d)
Memory usage: O(n*d)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from stackdb.filter import ChunkFilter
from stackdb.indexes.base_index import BaseIndex
from stackdb.models.chunk import Chunk


class FlatIndex(BaseIndex):
    def __init__(self, dimension: int):
        super().__init__(dimension)

    def add_vectors(self, chunks: List[Chunk]) -> None:
        self._verify_add_chunks(chunks)

        vectors = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        self.chunks.update({chunk.id: chunk for chunk in chunks})

        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.concatenate([self.vectors, vectors], axis=0)

        self.chunk_ids.extend([chunk.id for chunk in chunks])

    def remove_vectors(self, chunk_ids: List[str]) -> bool:
        if not all(cid in self.chunks for cid in chunk_ids):
            return False

        id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        indices_to_delete = sorted({id_to_idx[cid] for cid in chunk_ids}, reverse=True)

        self.vectors = np.delete(self.vectors, indices_to_delete, axis=0)

        self.chunk_ids = [cid for cid in self.chunk_ids if cid not in chunk_ids]
        self.chunks = {
            cid: chunk for cid, chunk in self.chunks.items() if cid not in chunk_ids
        }

        return True

    def search(
        self, query: List[float], k: int = 10, filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        if len(query) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query)} doesn't match index dimension {self.dimension}"
            )

        filterer = ChunkFilter(filter) if filter else None
        if self.vectors.shape[0] == 0:
            return []

        cid_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        query_array = np.array(query, dtype=np.float32)

        distances = []

        vectors = self.vectors
        chunk_ids = self.chunk_ids
        if filterer:
            chunk_ids = [
                chunk_id
                for chunk_id in chunk_ids
                if filterer.matches(self.chunks[chunk_id])
            ]
            indices = [cid_to_idx[chunk_id] for chunk_id in chunk_ids]
            vectors = vectors[indices]
        else:
            indices = [cid_to_idx[cid] for cid in chunk_ids]

        similarities = vectors @ query_array
        distances = 1.0 - similarities

        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[:k]

        results = []
        for idx in top_k_indices:
            chunk_id = chunk_ids[idx]
            distance = float(distances[idx])
            results.append((self.chunks[chunk_id], distance))

        return results

    def build_index(self) -> None:
        pass  # index is built incrementally

    def get_info(self) -> Dict[str, Any]:
        return {"type": "flat"}
