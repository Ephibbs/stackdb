"""
Vector Database Models Package

This package contains the hierarchical models for the vector database:
- Chunk: Text piece with embedding and metadata
- Document: Collection of chunks with metadata
- Library: Collection of documents with metadata
"""

from .chunk import Chunk, ChunkUpdate
from .document import Document, DocumentUpdate
from .library import Library, LibraryUpdate

__all__ = [
    "Chunk",
    "ChunkUpdate",
    "Document",
    "DocumentUpdate",
    "Library",
    "LibraryUpdate",
]
