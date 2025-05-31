"""
Chunk Model

A chunk represents a piece of text with an associated embedding vector and metadata.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid


class ChunkUpdate(BaseModel):
    id: str = Field(min_length=1)
    text: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict] = Field(None)


class Chunk(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(min_length=1)
    embedding: List[float] = Field(min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    document_id: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

    def update(self, **kwargs: Any):
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "metadata" in kwargs:
            self.metadata.update(kwargs["metadata"])
        if "document_id" in kwargs:
            self.document_id = kwargs["document_id"]
