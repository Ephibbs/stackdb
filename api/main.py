from fastapi import FastAPI, HTTPException, Query, status
from typing import List, Dict, Optional
import uvicorn
import argparse

from models import (
    BaseResponse,
    LibraryCreate,
    DocumentCreate,
    ChunkCreate,
    SearchQuery,
    SearchResult,
)
from stackdb.models import Library, LibraryUpdate, DocumentUpdate, ChunkUpdate
from api.helpers import get_search_results, get_library_responses, get_library_response, get_document_response, create_library_object
from api.persistence import load_libraries, save_libraries

libraries: Dict[str, Library] = {}
storage_path: Optional[str] = None

def get_library_object(library_id: str) -> Library:
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    return libraries[library_id]


app = FastAPI(
    title="Vector Database",
    description="A FastAPI-based vector database with libraries, documents, chunks, and k-NN search",
    version="0.0.1",
)

@app.on_event("startup")
async def startup_event():
    global libraries, storage_path
    if storage_path:
        libraries.update(load_libraries(storage_path))

@app.get("/")
async def root():
    return {
        "message": "Vector Database API is running",
        "total_libraries": len(libraries),
    }

@app.post("/libraries")
async def create_library(library_data: LibraryCreate):
    library = create_library_object(library_data, storage_path)

    libraries[library.id] = library
    
    if storage_path:
        save_libraries(libraries, storage_path)

    return BaseResponse(
        id=library.id,
        success=True,
        message=f"Library {library_data.name} created successfully",
    )

@app.get("/libraries")
async def list_libraries():
    return get_library_responses(libraries)

@app.get("/libraries/{library_id}")
async def get_library(library_id: str):
    library = get_library_object(library_id)
    return get_library_response(library)

@app.patch("/libraries/{library_id}")
async def update_library(library_id: str, update_data: LibraryUpdate):
    library = get_library_object(library_id)

    library.update(update_data)

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Library {library.name} updated successfully",
    )

@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str):
    if library_id not in libraries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
    del libraries[library_id]
    
    if storage_path:
        save_libraries(libraries, storage_path)
    
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Library {library_id} deleted successfully",
    )

# Document endpoints
@app.post("/libraries/{library_id}/documents")
async def create_documents(library_id: str, documents: List[DocumentCreate]):
    library = get_library_object(library_id)
    library.add_documents(documents)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Documents created successfully",
    )

@app.get("/libraries/{library_id}/documents")
async def list_documents(
    library_id: str,
    document_ids: Optional[List[str]] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    library = get_library_object(library_id)
    documents = library.get_documents(document_ids)

    results = []
    for document in documents[skip : skip + limit]:
        results.append(
            get_document_response(document)
        )

    return results

@app.get("/libraries/{library_id}/documents/{document_id}")
async def get_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    document = library.get_documents([document_id])[0]
    return get_document_response(document)

@app.patch("/libraries/{library_id}/documents/{document_id}")
async def update_document(
    library_id: str, document_id: str, update_data: DocumentUpdate
):
    library = get_library_object(library_id)
    library.update_document(document_id, update_data)
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} updated successfully",
    )

@app.delete("/libraries/{library_id}/documents/{document_id}")
async def delete_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    library.remove_document(document_id)
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} deleted successfully",
    )

# Chunk endpoints
@app.post("/libraries/{library_id}/chunks")
async def create_chunks(library_id: str, chunks: List[ChunkCreate]):
    library = get_library_object(library_id)
    library.add_chunks(chunks)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks created successfully",
    )

@app.get("/libraries/{library_id}/chunks")
async def list_chunks(
    library_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter: Optional[str] = Query(None),
):
    library = get_library_object(library_id)
    chunks = library.get_chunks(filter)
    return chunks[skip : skip + limit]

@app.get("/libraries/{library_id}/chunks/{chunk_id}")
async def get_chunk(library_id: str, chunk_id: str):
    library = get_library_object(library_id)
    chunk = library.get_chunks(id=chunk_id)[0]
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    return chunk

@app.patch("/libraries/{library_id}/chunks")
async def update_chunks(library_id: str, update_data: List[ChunkUpdate]):
    library = get_library_object(library_id)
    library.update_chunks(update_data)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks updated successfully",
    )

@app.delete("/libraries/{library_id}/chunks/{chunk_id}")
async def delete_chunks(library_id: str, chunk_ids: List[str]):
    library = get_library_object(library_id)
    library.remove_chunks(chunk_ids)

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks deleted successfully",
    )

# Index endpoints
@app.post("/libraries/{library_id}/search", response_model=List[SearchResult])
async def search_library(library_id: str, search_query: SearchQuery):
    library = get_library_object(library_id)

    results = get_search_results(library, search_query)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Vector Database API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--persistence-path",
        type=str,
        default="./data",
        help="Path where persistence data will be stored (default: ./data)",
    )

    args = parser.parse_args()

    storage_path = args.persistence_path

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Persistence path: {args.persistence_path}")

    uvicorn.run(app, host=args.host, port=args.port)
