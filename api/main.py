from fastapi import FastAPI
from typing import List
import uvicorn
import argparse

from models import (
    SearchQuery,
    SearchResult,
)
from api.helpers import get_search_results
from api.persistence import load_libraries, libraries, storage_path, get_library_object
from api.routes import library_crud, document_crud, chunk_crud

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

app.include_router(library_crud.router)
app.include_router(document_crud.router)
app.include_router(chunk_crud.router)

@app.post("/libraries/{library_id}/search", response_model=List[SearchResult])
def search_library(library_id: str, search_query: SearchQuery):
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
