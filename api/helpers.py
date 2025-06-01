from api.models import SearchResult, PartialChunkResponse, SearchQuery, LibraryResponse, DocumentResponse
from stackdb.models import Library, Document
from typing import List, Dict
from stackdb.indexes import get_index
from api.models import LibraryCreate
from pathlib import Path


def create_library_object(library_data: LibraryCreate, storage_path: str) -> Library:
    library_storage_path = None
    if storage_path:
        library_storage_path = Path(storage_path) / "libraries" / library_data.id
        library_storage_path.mkdir(parents=True, exist_ok=True)
    
    library = Library(
        id=library_data.id,
        name=library_data.name,
        dimension=library_data.dimension,
        metadata=library_data.metadata or {},
        storage_path=str(library_storage_path) if library_storage_path else None,
        index=get_index(
            library_data.index, library_data.dimension, library_data.index_params
        ),
    )
    return library

def get_library_responses(libraries: Dict[str, Library]) -> List[LibraryResponse]:
    results = []
    for library in libraries.values():
        results.append(
            get_library_response(library)
        )
    return results

def get_library_response(library: Library) -> LibraryResponse:
    return LibraryResponse(
                id=library.id,
                name=library.name,
                metadata=library.metadata,
                index_information=library.index.get_information(),
                document_ids=list(library.documents.keys()),
                created_at=library.created_at.isoformat(),
            )

def get_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        metadata=document.metadata,
        library_id=document.library_id,
        chunk_count=len(document.chunks),
        created_at=document.created_at.isoformat(),
    )


def get_search_results(library: Library, search_query: SearchQuery) -> List[SearchResult]:
    search_results = library.search_chunks(
        query=search_query.query,
        k=search_query.k,
        filter=search_query.filter,
    )
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