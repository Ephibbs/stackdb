"""
Library Model

A library represents a collection of documents with associated metadata.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any, Tuple, Callable
import uuid
from datetime import datetime
from stackdb.models.document import Document, DocumentCreate
from stackdb.models.chunk import Chunk, ChunkUpdate
from stackdb.filter import ChunkFilter
from stackdb.indexes import BaseIndex, FlatIndex, get_index
from stackdb.lock import library_write_lock
from stackdb.persistence import PersistenceManager
import functools


def persistence_logger(operation: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            operation_data = {
                "args": args,
                "kwargs": kwargs,
            }

            if self.persistence_manager:
                self.persistence_manager.log(operation, **operation_data)

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class Library(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    documents: Optional[Dict[str, Document]] = Field(default_factory=dict)
    index: BaseIndex = Field(default=None)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    storage_path: Optional[str] = Field(default=None)
    persistence_manager: Optional[PersistenceManager] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        if self.index is None:
            self.index = FlatIndex(dimension=self.dimension)

        if self.storage_path:
            self.persistence_manager = PersistenceManager(self.storage_path)
            self._restore_from_persistence()
            self.persistence_manager.set_snapshot_callback(self._to_snapshot_data)

    def _restore_from_persistence(self) -> None:
        if not self.persistence_manager:
            return

        restore_data = self.persistence_manager.restore_library_state()
        if not restore_data:
            return

        snapshot_data = restore_data.get("snapshot_data")
        if snapshot_data:
            self._restore_from_snapshot(snapshot_data)

        wal_entries = restore_data.get("wal_entries", [])
        last_wal_sequence = restore_data.get("last_wal_sequence", 0)

        for entry in wal_entries:
            if entry.sequence > last_wal_sequence:
                self._apply_wal_entry(entry)

    def _restore_from_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        if "name" in snapshot_data:
            self.name = snapshot_data["name"]
        if "dimension" in snapshot_data:
            self.dimension = snapshot_data["dimension"]
        if "metadata" in snapshot_data:
            self.metadata = snapshot_data["metadata"]
        if "created_at" in snapshot_data:
            self.created_at = datetime.fromisoformat(snapshot_data["created_at"])

        if "documents" in snapshot_data:
            documents_data = snapshot_data["documents"]
            self.documents = {}
            for doc_id, doc_data in documents_data.items():
                self.documents[doc_id] = Document(**doc_data)

        if "index" in snapshot_data:
            index_data = snapshot_data["index"]
            self.index = get_index(
                index_data["type"], self.dimension, index_data.get("params", {})
            )

            all_chunks = []
            for document in self.documents.values():
                all_chunks.extend(document.chunks.values())

            if all_chunks:
                self.index.add_vectors(all_chunks)

    def _apply_wal_entry(self, entry) -> None:
        operation = entry.operation
        data = entry.kwargs

        try:
            if operation == "add_documents":
                documents = [Document(**doc_data) for doc_data in data["documents"]]
                self._add_documents_internal(documents)
            elif operation == "remove_documents":
                self._remove_documents_internal(data["document_ids"])
            elif operation == "update_document":
                self._update_document_internal(
                    data["document_id"], **data["update_data"]
                )
            elif operation == "add_chunks":
                chunks = [Chunk(**chunk_data) for chunk_data in data["chunks"]]
                self._add_chunks_internal(chunks)
            elif operation == "remove_chunks":
                self._remove_chunks_internal(data["chunk_ids"])
            elif operation == "update_chunks":
                self._update_chunks_internal(data["chunk_ids"], data["update_data"])
            elif operation == "update":
                self._update_internal(**data)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            print(
                f"Error applying WAL entry {operation} (sequence {entry.sequence}): {e}"
            )

    def _to_snapshot_data(self) -> Dict[str, Any]:
        documents_data = {}
        for doc_id, doc in self.documents.items():
            documents_data[doc_id] = doc.model_dump()

        index_info = self.index.get_info()

        return {
            "id": self.id,
            "name": self.name,
            "dimension": self.dimension,
            "metadata": self.metadata,
            "documents": documents_data,
            "index": index_info,
            "created_at": self.created_at.isoformat(),
        }

    """
    Document Methods
    """

    @persistence_logger("add_documents")
    def add_documents(self, documents: List[Document]):
        self._add_documents_internal(documents)

    @library_write_lock
    def _add_documents_internal(self, documents: List[Document]):
        for document in documents:
            if document.id in self.documents:
                raise ValueError(f"Document {document.id} already exists")

        new_chunks = []
        for document in documents:
            self.documents[document.id] = document
            new_chunks.extend(document.chunks.values())

        self.index.add_vectors(new_chunks)

    @persistence_logger("remove_documents")
    def remove_documents(self, document_ids: List[str]):
        self._remove_documents_internal(document_ids)

    @library_write_lock
    def _remove_documents_internal(self, document_ids: List[str]):
        removed_chunk_ids = []
        for document_id in document_ids:
            if document_id in self.documents:
                document = self.documents.pop(document_id)
                removed_chunk_ids.extend(document.chunks.keys())
        self.index.remove_vectors(removed_chunk_ids)

    def get_documents(self, document_ids: Optional[List[str]] = None) -> List[Document]:
        if document_ids is None:
            return list(self.documents.values())
        return [
            self.documents[document_id]
            for document_id in document_ids
            if document_id in self.documents
        ]

    @persistence_logger("update_document")
    def update_document(self, document_id: str, **kwargs: Any):
        self._update_document_internal(document_id, **kwargs)

    def _update_document_internal(self, document_id: str, **kwargs: Any):
        if document_id not in self.documents:
            raise ValueError(f"Document {document_id} not found")
        document = self.documents[document_id]
        document.update(**kwargs)

    """
    Chunk Methods
    """

    @persistence_logger("add_chunks")
    def add_chunks(self, chunks: List[Chunk]):
        self._add_chunks_internal(chunks)

    @library_write_lock
    def _add_chunks_internal(self, chunks: List[Chunk]):
        for chunk in chunks:
            if chunk.document_id not in self.documents:
                raise ValueError(f"Document {chunk.document_id} not found")
        for chunk in chunks:
            document = self.documents[chunk.document_id]
            document.chunks[chunk.id] = chunk
        self.index.add_vectors(chunks)

    @persistence_logger("remove_chunks")
    def remove_chunks(self, chunk_ids: List[str]):
        self._remove_chunks_internal(chunk_ids)

    @library_write_lock
    def _remove_chunks_internal(self, chunk_ids: List[str]):
        for chunk_id in chunk_ids:
            chunk = self.index.chunks[chunk_id]
            document = self.documents[chunk.document_id]
            document.chunks.pop(chunk_id)
            if len(document.chunks) == 0:
                del self.documents[document.id]
        self.index.remove_vectors(chunk_ids)

    def get_chunks(self, filter: Optional[Dict] = None) -> List[Chunk]:
        if filter is None:
            return list(self.index.chunks.values())
        filterer = ChunkFilter(filter)
        return filterer.filter_chunks(self.index.chunks.values())

    def search_chunks(
        self, query: List[float], k: int, filter: Optional[Dict] = None, **kwargs: Any
    ) -> List[Tuple[str, float]]:
        return self.index.search(query, k, filter, **kwargs)

    @persistence_logger("update_chunks")
    def update_chunks(self, update_data: List[ChunkUpdate]):
        self._update_chunks_internal(update_data)

    def _update_chunks_internal(self, update_data: List[ChunkUpdate]):
        for chunk_update in update_data:
            if chunk_update.id not in self.index.chunks:
                raise ValueError(f"Chunk {chunk_update.id} not found")
            chunk = self.index.chunks[chunk_update.id]
            chunk.update(**chunk_update.model_dump())

    @persistence_logger("update")
    def update(self, **kwargs: Any):
        self._update_internal(kwargs)

    def _update_internal(self, kwargs: Dict[str, Any]):
        for key, value in kwargs.items():
            if key == "index":
                all_chunks = list(self.index.chunks.values())
                del self.index
                self.index = get_index(
                    value["type"], self.dimension, value.get("params", {})
                )
                self.index.add_vectors(all_chunks)
            else:
                setattr(self, key, value)

    """
    Persistence Methods
    """

    def create_snapshot(self, snapshot_id: Optional[str] = None) -> Optional[str]:
        if not self.persistence_manager:
            return None

        snapshot_data = self._to_snapshot_data()
        return self.persistence_manager.create_snapshot(snapshot_data, snapshot_id)

    def flush(self) -> None:
        if self.persistence_manager:
            self.persistence_manager.flush()
