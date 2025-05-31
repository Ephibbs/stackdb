"""
StackDB - Vector Database
version: 0.1.0
"""

from .models import Chunk, Document, Library
from .indexes import BaseIndex, FlatIndex, IVFIndex, LSHIndex
from .persistence import PersistenceManager, BufferedWAL, SnapshotManager, WALEntry

__all__ = [
    # Models
    "Chunk",
    "Document",
    "Library",
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
