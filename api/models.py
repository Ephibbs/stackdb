from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uuid


class BaseResponse(BaseModel):
    id: str
    success: bool
    message: str


# Library endpoints
class LibraryCreate(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    metadata: Optional[Dict] = Field(default_factory=dict)
    index: Optional[str] = Field(default="flat", enum=["flat", "lsh", "ivf"])
    index_params: Optional[Dict] = Field(default_factory=dict)


class LibraryResponse(BaseModel):
    id: str
    name: str
    metadata: Dict
    index_information: Dict
    document_ids: List[str]
    created_at: str


class LibraryUpdate(BaseModel):
    id: str = Field(min_length=1)
    name: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict] = Field(None)


# Document endpoints
class DocumentChunk(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(min_length=1)
    embedding: List[float] = Field()
    metadata: Optional[Dict] = Field(default_factory=dict)


class DocumentCreate(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict] = Field(default_factory=dict)
    chunks: Optional[List[DocumentChunk]] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    id: str
    title: str
    metadata: Dict
    library_id: str
    chunk_count: int
    created_at: str


class DocumentUpdate(BaseModel):
    id: str = Field(min_length=1)
    title: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict] = Field(None)


# Chunk endpoints
class ChunkCreate(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(min_length=1)
    embedding: List[float] = Field()
    metadata: Optional[Dict] = Field(default_factory=dict)
    document_id: Optional[str] = Field(None)


class PartialChunkResponse(BaseModel):
    id: Optional[str] = Field(None)
    text: Optional[str] = Field(None)
    embedding: Optional[List[float]] = Field(None)
    metadata: Optional[Dict] = Field(None)
    document_id: Optional[str] = Field(None)
    created_at: Optional[datetime] = Field(None)


class ChunkUpdate(BaseModel):
    id: str = Field(min_length=1)
    text: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict] = Field(None)


# Search types
class SearchQuery(BaseModel):
    query: List[float] = Field()
    k: int = Field(default=10, ge=1, le=100)
    filter: Optional[str] = Field(None)
    fields: Optional[List[str]] = Field(None)


class SearchResult(BaseModel):
    distance: float
    chunk: PartialChunkResponse
