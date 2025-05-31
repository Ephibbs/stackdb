"""
Document Model

A document represents a collection of chunks with associated metadata.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
import uuid
from .chunk import Chunk


class DocumentCreate(BaseModel):
    title: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chunks: Optional[List[Chunk]] = Field(default_factory=list)


class Document(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    library_id: Optional[str] = Field(default=None)
    chunks: Optional[Dict[str, Chunk]] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

    def update(self, **kwargs: Any):
        if "title" in kwargs:
            self.title = kwargs["title"]
        if "metadata" in kwargs:
            self.metadata.update(kwargs["metadata"])
        if "library_id" in kwargs:
            self.library_id = kwargs["library_id"]

    def add_chunk(self, chunk: Chunk):
        self.chunks[chunk.id] = chunk
        chunk.document_id = self.id

    def remove_chunk(self, chunk_id: str):
        del self.chunks[chunk_id]
