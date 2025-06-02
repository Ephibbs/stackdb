from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uuid
import json

Metadata = Dict[str, str | int | float | bool | None]
MAX_EMBEDDING_DIMENSION = 16384


class BaseResponse(BaseModel):
    id: str
    success: bool
    message: str


# Library endpoints
class LibraryCreate(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=255)
    dimension: int = Field(ge=1, le=MAX_EMBEDDING_DIMENSION)
    metadata: Optional[Metadata] = Field(default_factory=dict)
    index: Optional[str] = Field(default="flat", pattern="^(flat|lsh|ivf)$")
    index_params: Optional[Metadata] = Field(default_factory=dict)


class LibraryResponse(BaseModel):
    id: str
    name: str
    metadata: Dict
    index_information: Dict
    document_ids: List[str]
    created_at: str


# Document endpoints
class DocumentCreate(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Metadata] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    id: str
    title: str
    metadata: Dict
    library_id: str
    chunk_count: int
    created_at: str


# Chunk endpoints
class PartialChunkResponse(BaseModel):
    id: Optional[str] = Field(None)
    text: Optional[str] = Field(None, max_length=50000)
    embedding: Optional[List[float]] = Field(None)
    metadata: Optional[Metadata] = Field(None)
    document_id: Optional[str] = Field(None)
    created_at: Optional[datetime] = Field(None)


# Search types
class SearchQuery(BaseModel):
    query: List[float] = Field(min_length=1, max_length=MAX_EMBEDDING_DIMENSION)
    k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[str] = Field(None, max_length=1000)
    fields: Optional[List[str]] = Field(None, max_length=50)


class SearchResult(BaseModel):
    distance: float
    chunk: PartialChunkResponse
