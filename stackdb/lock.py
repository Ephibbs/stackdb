from __future__ import annotations
from typing import Callable, Dict, Any
import threading
import functools

__all__ = [
    "library_write_lock",
    "library_read_lock",
]

_library_locks: Dict[str, ReadWriteLock] = {}
_create_library_lock = threading.Lock()


class ReadWriteLock:
    def __init__(self):
        self._read_count = 0
        self._writer_waiting = 0
        self._read_count_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader_entry_lock = threading.Lock()

    def acquire_read(self):
        with self._reader_entry_lock:
            with self._read_count_lock:
                self._read_count += 1
                if self._read_count == 1:
                    self._write_lock.acquire()

    def release_read(self):
        with self._read_count_lock:
            self._read_count -= 1
            if self._read_count == 0:
                self._write_lock.release()

    def acquire_write(self):
        with self._read_count_lock:
            self._writer_waiting += 1
        self._reader_entry_lock.acquire()
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()
        with self._read_count_lock:
            self._writer_waiting -= 1
        self._reader_entry_lock.release()


def _get_lock(library_id: str) -> ReadWriteLock:
    lock = _library_locks.get(library_id)
    if lock is not None:
        return lock

    with _create_library_lock:
        lock = _library_locks.get(library_id)
        if lock is None:
            lock = ReadWriteLock()
            _library_locks[library_id] = lock
        return lock


def library_read_lock(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any):
        library_id = getattr(self, "id", None)
        if library_id is None:
            raise AttributeError(
                "library_read_lock decorated methods require the instance to "
                "have an 'id' attribute"
            )

        lock = _get_lock(str(library_id))
        lock.acquire_read()
        try:
            return func(self, *args, **kwargs)
        finally:
            lock.release_read()

    return wrapper


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
        lock.acquire_write()
        try:
            return func(self, *args, **kwargs)
        finally:
            lock.release_write()

    return wrapper
