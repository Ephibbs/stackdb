"""
Abstract base class that defines the interface for all vector index implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from stackdb.models.chunk import Chunk


class BaseIndex(ABC):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunks: Dict[str, Chunk] = {}
        self.vectors: np.ndarray = np.empty((0, dimension))
        self.chunk_ids: List[str] = []

    def _verify_add_chunks(self, chunks: List[Chunk]) -> bool:
        for chunk in chunks:
            if len(chunk.embedding) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(chunk.embedding)} doesn't match index dimension {self.dimension}"
                )

        incoming_chunk_ids = set(
            [chunk.id for chunk in chunks if chunk.id not in self.chunks]
        )
        if len(incoming_chunk_ids) != len(chunks):
            raise ValueError("Duplicate chunk ID found")

        return True

    @abstractmethod
    def add_vectors(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def remove_vectors(self, chunk_ids: List[str]) -> bool:
        pass

    @abstractmethod
    def search(
        self, query: List[float], k: int = 10, filter: Optional[str] = None, **kwargs
    ) -> List[Tuple[str, float]]:
        """
        query: normalized query vector to search for
        filter: filter to apply to the search ex. "metadata.key = value and created_at > 2025-01-01"
        """
        pass

    @abstractmethod
    def build_index(self) -> None:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    def get_vector_count(self) -> int:
        return len(self.chunk_ids)

    def get_dimension(self) -> int:
        return self.dimension
