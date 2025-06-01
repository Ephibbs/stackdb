from fastapi import APIRouter, Query
from api.models import DocumentCreate, BaseResponse
from api.helpers import get_document_response, convert_document_create_to_document
from api.persistence import get_library_object
from typing import List, Optional
from stackdb.models import DocumentUpdate

router = APIRouter(prefix="/libraries/{library_id}/documents")


@router.post("")
def create_documents(library_id: str, documents: List[DocumentCreate]):
    library = get_library_object(library_id)
    converted_documents = convert_document_create_to_document(documents, library_id)
    library.add_documents(converted_documents)
    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Documents created successfully",
    )


@router.get("")
def list_documents(
    library_id: str,
    document_ids: Optional[List[str]] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    library = get_library_object(library_id)
    documents = library.get_documents(document_ids)

    results = []
    for document in documents[skip : skip + limit]:
        results.append(get_document_response(document))

    return results


@router.get("/{document_id}")
def get_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    document = library.get_documents([document_id])[0]
    return get_document_response(document)


@router.patch("/{document_id}")
def update_document(library_id: str, document_id: str, update_data: DocumentUpdate):
    library = get_library_object(library_id)
    update_data.id = document_id
    library.update_document(update_data)
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} updated successfully",
    )


@router.delete("/{document_id}")
def delete_document(library_id: str, document_id: str):
    library = get_library_object(library_id)
    library.remove_document(document_id)
    return BaseResponse(
        id=document_id,
        success=True,
        message=f"Document {document_id} deleted successfully",
    )
