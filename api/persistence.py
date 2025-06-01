from typing import Dict
from pathlib import Path
import json
from stackdb.models import Library
from fastapi import HTTPException, status
from typing import Optional

storage_path: Optional[str] = None
libraries: Dict[str, Library] = {}


def get_library_object(library_id: str) -> Library:
    if library_id not in libraries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Library not found"
        )
    return libraries[library_id]


def save_libraries(libraries: Dict[str, Library], storage_path: str) -> None:
    storage_dir = Path(storage_path)
    storage_dir.mkdir(parents=True, exist_ok=True)

    library_ids = list(libraries.keys())

    with open(storage_dir / "library_registry.json", "w") as f:
        json.dump(library_ids, f, indent=2)


def load_libraries(storage_path: str) -> Dict[str, Library]:
    storage_dir = Path(storage_path)
    registry_file = storage_dir / "library_registry.json"

    if not registry_file.exists():
        return {}

    try:
        with open(registry_file, "r") as f:
            library_ids = json.load(f)

        libraries = {}
        for lib_id in library_ids:
            library_storage_path = storage_dir / "libraries" / lib_id
            library_storage_path.mkdir(parents=True, exist_ok=True)

            library = Library.restore_from_persistence(
                id=lib_id, storage_path=str(library_storage_path)
            )

            libraries[lib_id] = library

        return libraries
    except Exception:
        return {}
