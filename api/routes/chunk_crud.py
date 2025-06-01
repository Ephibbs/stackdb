from fastapi import APIRouter, Query
from api.models import ChunkCreate, BaseResponse
from api.persistence import get_library_object
from typing import List, Optional
from stackdb.models import ChunkUpdate
from fastapi import HTTPException, status

router = APIRouter(prefix="/libraries/{library_id}/chunks")

@router.post("")
def create_chunks(library_id: str, chunks: List[ChunkCreate]):
    library = get_library_object(library_id)
    library.add_chunks(chunks)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks created successfully",
    )

@router.get("")
def list_chunks(
    library_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter: Optional[str] = Query(None),
):
    library = get_library_object(library_id)
    chunks = library.get_chunks(filter)
    return chunks[skip : skip + limit]

@router.get("/{chunk_id}")
def get_chunk(library_id: str, chunk_id: str):
    library = get_library_object(library_id)
    chunk = library.get_chunks(id=chunk_id)[0]
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    return chunk

@router.patch("")
def update_chunks(library_id: str, update_data: List[ChunkUpdate]):
    library = get_library_object(library_id)
    library.update_chunks(update_data)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks updated successfully",
    )

@router.delete("/{chunk_id}")
def delete_chunks(library_id: str, chunk_ids: List[str]):
    library = get_library_object(library_id)
    library.remove_chunks(chunk_ids)

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks deleted successfully",
    )
