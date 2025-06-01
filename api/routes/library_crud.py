from fastapi import APIRouter, HTTPException, status
from api.models import LibraryCreate, LibraryUpdate, BaseResponse
from api.helpers import create_library_object, get_library_responses, get_library_response
from api.persistence import save_libraries, libraries, storage_path, get_library_object

router = APIRouter(prefix="/libraries")

@router.post("")
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

@router.get("")
async def list_libraries():
    return get_library_responses(libraries)

@router.get("/{library_id}")
async def get_library(library_id: str):
    library = get_library_object(library_id)
    return get_library_response(library)

@router.patch("/{library_id}")
async def update_library(library_id: str, update_data: LibraryUpdate):
    library = get_library_object(library_id)

    library.update(update_data)

    return BaseResponse(
        id=library_id,
        success=True,
        message=f"Library {library.name} updated successfully",
    )

@router.delete("/{library_id}")
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
