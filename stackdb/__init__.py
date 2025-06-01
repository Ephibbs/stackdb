"""
StackDB - Vector Database
version: 0.1.0
"""

from .models import Chunk, Document, Library, ChunkUpdate, DocumentUpdate, LibraryUpdate
from .indexes import BaseIndex, FlatIndex, IVFIndex, LSHIndex
from .persistence import PersistenceManager, BufferedWAL, SnapshotManager, WALEntry

__all__ = [
    # Models
    "Chunk",
    "ChunkUpdate",
    "Document",
    "DocumentUpdate",
    "Library",
    "LibraryUpdate",
    # Indexes
    "BaseIndex",
    "FlatIndex",
    "IVFIndex",
    "LSHIndex",
    # Persistence
    "PersistenceManager",
    "BufferedWAL",
    "SnapshotManager",
    "WALEntry",
]

__version__ = "0.1.0"
