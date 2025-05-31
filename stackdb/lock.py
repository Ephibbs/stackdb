from __future__ import annotations
from typing import Callable, Dict, Any
import threading
import functools

__all__ = [
    "library_write_lock",
]

_library_locks: Dict[str, threading.RLock] = {}
_create_library_lock = threading.Lock()


def _get_lock(library_id: str) -> threading.RLock:
    lock = _library_locks.get(library_id)
    if lock is not None:
        return lock

    with _create_library_lock:
        lock = _library_locks.get(library_id)
        if lock is None:
            lock = threading.RLock()
            _library_locks[library_id] = lock
        return lock


def library_write_lock(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any):
        library_id = getattr(self, "id", None)
        if library_id is None:
            raise AttributeError(
                "library_write_lock decorated methods require the instance to "
                "have an 'id' attribute"
            )

        lock = _get_lock(str(library_id))
        with lock:
            return func(self, *args, **kwargs)

    return wrapper
