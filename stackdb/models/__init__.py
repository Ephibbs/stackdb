"""
Vector Database Models Package

This package contains the hierarchical models for the vector database:
- Chunk: Text piece with embedding and metadata
- Document: Collection of chunks with metadata
- Library: Collection of documents with metadata
"""

from .chunk import Chunk, ChunkUpdate
from .document import Document
from .library import Library

__all__ = ["Chunk", "ChunkUpdate", "Document", "Library"]
