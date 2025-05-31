from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Optional
import uvicorn
import argparse

from models import (
    BaseResponse,
    LibraryCreate,
    LibraryResponse,
    LibraryUpdate,
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    ChunkCreate,
    PartialChunkResponse,
    SearchQuery,
    SearchResult,
    ChunkUpdate,
)
from stackdb.models import Library, Document, Chunk
from stackdb.indexes import get_index

# In-memory storage
libraries: Dict[str, Library] = {}

app = FastAPI(
    title="Hierarchical Vector Database",
    description="A FastAPI-based hierarchical vector database with libraries, documents, chunks, and k-NN search",
    version="0.0.1",
)


def get_library_object(library_id: str) -> Library:
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    return libraries[library_id]


# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Vector Database API is running",
        "total_libraries": len(libraries),
    }


# Library endpoints
@app.post("/libraries", response_model=BaseResponse)
async def create_library(library_data: LibraryCreate):
    library = Library(
        name=library_data.name,
        dimension=library_data.dimension,
        metadata=library_data.metadata or {},
        index=get_index(
            library_data.index, library_data.dimension, library_data.index_params
        ),
    )

    libraries[library.id] = library

    return BaseResponse(
        id=library.id,
        success=True,
        message=f"Library {library_data.name} created successfully",
    )


@app.get("/libraries", response_model=List[LibraryResponse])
async def list_libraries():
    results = []
    for lib_id, library in libraries.items():
        results.append(
            LibraryResponse(
                id=lib_id,
                name=library.name,
                metadata=library.metadata,
                index_information=library.index.get_information(),
                document_ids=list(library.documents.keys()),
                created_at=library.created_at.isoformat(),
            )
        )
    return results


@app.get("/libraries/{library_id}", response_model=LibraryResponse)
async def get_library(library_id: str):
    library = get_library_object(library_id)
    return LibraryResponse(
        id=library_id,
        name=library.name,
        metadata=library.metadata,
        document_ids=list(library.documents.keys()),
        index_information=library.index.get_information(),
        created_at=library.created_at.isoformat(),
    )


@app.patch("/libraries/{library_id}", response_model=BaseResponse)
async def update_library(library_id: str, update_data: LibraryUpdate):
    library = get_library_object(library_id)

    library.update(**update_data.model_dump())

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Library {library.name} updated successfully",
    )


@app.delete("/libraries/{library_id}", response_model=BaseResponse)
async def delete_library(library_id: str):
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    del libraries[library_id]
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Library {library_id} deleted successfully",
    )


# Document endpoints
@app.post("/libraries/{library_id}/documents", response_model=BaseResponse)
async def create_documents(library_id: str, documents: List[DocumentCreate]):
    library = get_library_object(library_id)
    documents = [
        Document(**document.model_dump(exclude={"chunks"})) for document in documents
    ]
    for document in documents:
        for chunk in document.chunks:
            chunk = Chunk(**chunk.model_dump())
            chunk.document_id = document.id
            document.add_chunk(chunk)
    library.add_documents(documents)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Documents created successfully",
    )


@app.get("/libraries/{library_id}/documents", response_model=List[DocumentResponse])
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
            DocumentResponse(
                id=document.id,
                title=document.title,
                metadata=document.metadata,
                library_id=library_id,
                chunk_count=len(document.chunks),
                created_at=document.created_at.isoformat(),
            )
        )

    return results


@app.get(
    "/libraries/{library_id}/documents/{document_id}", response_model=DocumentResponse
)
async def get_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    document = library.get_documents([document_id])[0]
    return DocumentResponse(
        id=document.id,
        title=document.title,
        metadata=document.metadata,
        library_id=library_id,
        chunk_ids=[chunk.id for chunk in document.chunks],
        chunk_count=len(document.chunks),
        created_at=document.created_at.isoformat(),
    )


@app.patch(
    "/libraries/{library_id}/documents/{document_id}",
    response_model=BaseResponse,
)
async def update_document(
    library_id: str, document_id: str, update_data: DocumentUpdate
):
    library = get_library_object(library_id)
    library.update_document(document_id, **update_data.model_dump())
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} updated successfully",
    )


@app.delete(
    "/libraries/{library_id}/documents/{document_id}", response_model=BaseResponse
)
async def delete_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    library.remove_document(document_id)
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} deleted successfully",
    )


# Chunk endpoints
@app.post("/libraries/{library_id}/chunks", response_model=BaseResponse)
async def create_chunks(library_id: str, chunks: List[ChunkCreate]):
    library = get_library_object(library_id)
    library.add_chunks([Chunk(**chunk.model_dump()) for chunk in chunks])
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks created successfully",
    )


@app.get("/libraries/{library_id}/chunks", response_model=List[Chunk])
async def list_chunks(
    library_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter: Optional[str] = Query(None),
):
    library = get_library_object(library_id)
    chunks = library.get_chunks(filter)
    return chunks[skip : skip + limit]


@app.get("/libraries/{library_id}/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(library_id: str, chunk_id: str):
    library = get_library_object(library_id)
    chunks = library.get_chunks(f'id = "{chunk_id}"')
    if len(chunks) == 0:
        raise HTTPException(status_code=404, detail="Chunk not found")
    chunk = chunks[0]
    return chunk


@app.patch("/libraries/{library_id}/chunks", response_model=BaseResponse)
async def update_chunks(library_id: str, update_data: List[ChunkUpdate]):
    library = get_library_object(library_id)
    library.update_chunks(update_data)
    chunk_ids = [chunk.id for chunk in update_data]
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks {chunk_ids} updated successfully",
    )


@app.delete("/libraries/{library_id}/chunks/{chunk_id}", response_model=BaseResponse)
async def delete_chunks(library_id: str, chunk_ids: List[str]):
    library = get_library_object(library_id)
    library.remove_chunks(chunk_ids)

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Chunks {chunk_ids} deleted successfully",
    )


# Index endpoints
@app.post("/libraries/{library_id}/search", response_model=List[SearchResult])
async def search_library(library_id: str, search_query: SearchQuery):
    library = get_library_object(library_id)

    try:
        search_results = library.search_chunks(
            query=search_query.query,
            k=search_query.k,
            filter=search_query.filter,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")

    results = []
    for chunk, distance in search_results:
        if search_query.fields:
            chunk_response = {
                field: getattr(chunk, field) for field in search_query.fields
            }
        else:
            chunk_response = chunk
        chunk_response = PartialChunkResponse(**chunk_response.model_dump())
        results.append(SearchResult(distance=distance, chunk=chunk_response))

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

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Persistence path: {args.persistence_path}")

    uvicorn.run(app, host=args.host, port=args.port)
