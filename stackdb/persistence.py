"""
Persistence Module for StackDB

Implements Write-Ahead Logging (WAL) and snapshot functionality for persistent storage.

WAL could be used with Kafka for a more robust or distributed system.
"""

import json
import pickle
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import gzip


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Pydantic models."""

    def default(self, obj):
        if hasattr(obj, "model_dump"):  # Pydantic model
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def decode_json_object(obj):
    """Standalone object hook function for JSON decoding."""
    if isinstance(obj, dict) and "id" in obj:
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class WALEntry:
    def __init__(
        self,
        operation: str,
        args: Any,
        kwargs: Dict[str, Any],
        sequence: int,
        timestamp: Optional[datetime] = None,
    ):
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.sequence = sequence
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "args": self.args,
            "kwargs": self.kwargs,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WALEntry":
        return cls(
            operation=data["operation"],
            args=data["args"],
            kwargs=data["kwargs"],
            sequence=data["sequence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class DurableCounter:
    def __init__(self, counter_path: str):
        """
        counter_path: str path to the counter file .txt
        A durable counter that can be used to track the sequence of operations.
        """
        self.counter_path = Path(counter_path)
        self.counter_lock = threading.Lock()
        self.counter = self._load_counter()

    def _load_counter(self) -> int:
        if self.counter_path.exists():
            try:
                with open(self.counter_path, "r") as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                pass
        return 0

    def _save_counter(self) -> None:
        with open(self.counter_path, "w") as f:
            f.write(str(self.counter))

    def increment(self) -> int:
        with self.counter_lock:
            self.counter += 1
            self._save_counter()
            return self.counter

    def get_current_value(self) -> int:
        return self.counter

    def reset(self) -> None:
        with self.counter_lock:
            self.counter = 0
            self._save_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()


class BufferedWAL:
    def __init__(
        self, storage_path: str, buffer_size: int = 100, flush_interval: float = 5.0
    ):
        self.storage_path = Path(storage_path)
        self.wal_path = self.storage_path / "wal.log"
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.buffer: List[WALEntry] = []
        self.buffer_lock = threading.Lock()
        self.last_flush = time.time()

        self.sequence_counter = DurableCounter(self.storage_path / "wal_sequence.txt")

        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()

    def write_entry(self, operation: str, args: Any, kwargs: Any) -> int:
        sequence = self.sequence_counter.increment()
        entry = WALEntry(operation, args, kwargs, sequence)

        with self.buffer_lock:
            self.buffer.append(entry)

            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()

        return sequence

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return

        with open(self.wal_path, "a", encoding="utf-8") as f:
            for entry in self.buffer:
                f.write(json.dumps(entry.to_dict(), cls=JSONEncoder) + "\n")

        self.buffer.clear()
        self.last_flush = time.time()

    def flush(self) -> None:
        with self.buffer_lock:
            self._flush_buffer()

    def _background_flush(self) -> None:
        while True:
            time.sleep(self.flush_interval)
            current_time = time.time()

            with self.buffer_lock:
                if (
                    self.buffer
                    and (current_time - self.last_flush) >= self.flush_interval
                ):
                    self._flush_buffer()

    def read_entries(self, min_sequence: int = 0) -> List[WALEntry]:
        if not self.wal_path.exists():
            return []
        entries = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry_data = json.loads(line, object_hook=decode_json_object)
                        entry = WALEntry.from_dict(entry_data)
                        if entry.sequence > min_sequence:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        return entries

    def get_current_sequence(self) -> int:
        return self.sequence_counter.get_current_value()

    def clear(self) -> None:
        with self.buffer_lock:
            self.buffer.clear()
            if self.wal_path.exists():
                self.wal_path.unlink()

    def cleanup_old_entries(self, min_sequence_to_keep: int) -> None:
        if not self.wal_path.exists():
            return

        all_entries = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry_data = json.loads(line, object_hook=decode_json_object)
                        entry = WALEntry.from_dict(entry_data)
                        if entry.sequence >= min_sequence_to_keep:
                            all_entries.append(entry)
                    except json.JSONDecodeError:
                        continue

        with self.buffer_lock:
            self._flush_buffer()

            with open(self.wal_path, "w", encoding="utf-8") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry.to_dict(), cls=JSONEncoder) + "\n")

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SnapshotManager:
    def __init__(self, storage_path: str, snapshot_interval: float = 300.0):
        self.storage_path = Path(storage_path)
        self.snapshots_path = self.storage_path / "snapshots"
        self.snapshot_interval = snapshot_interval
        self._snapshot_callback = None

        self.snapshots_path.mkdir(parents=True, exist_ok=True)

        self._snapshot_thread = None
        self._stop_snapshot_thread = threading.Event()

        self._start_snapshot_thread()

    def set_snapshot_callback(
        self, callback: Callable[[], tuple[Dict[str, Any], int]]
    ) -> None:
        self._snapshot_callback = callback

    def _start_snapshot_thread(self) -> None:
        if self._snapshot_thread is None or not self._snapshot_thread.is_alive():
            self._stop_snapshot_thread.clear()
            self._snapshot_thread = threading.Thread(
                target=self._background_snapshot, daemon=True
            )
            self._snapshot_thread.start()

    def _background_snapshot(self) -> None:
        while not self._stop_snapshot_thread.is_set():
            try:
                if self._stop_snapshot_thread.wait(self.snapshot_interval):
                    break

                if self._snapshot_callback is not None:
                    try:
                        library_data, last_wal_sequence = self._snapshot_callback()
                        self.create_snapshot(library_data, last_wal_sequence)
                    except Exception:
                        pass

            except Exception:
                pass

    def stop_snapshot_thread(self) -> None:
        if self._snapshot_thread is not None:
            self._stop_snapshot_thread.set()
            self._snapshot_thread.join(timeout=10.0)

    def create_snapshot(
        self,
        library_data: Dict[str, Any],
        last_wal_sequence: int,
        snapshot_id: Optional[str] = None,
    ) -> str:
        if snapshot_id is None:
            snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        snapshot_path = self.snapshots_path / f"snapshot_{snapshot_id}.pkl.gz"

        snapshot_data = {
            "library_data": library_data,
            "last_wal_sequence": last_wal_sequence,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(),
        }

        with gzip.open(snapshot_path, "wb") as f:
            pickle.dump(snapshot_data, f)

        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        snapshot_path = self.snapshots_path / f"snapshot_{snapshot_id}.pkl.gz"

        if not snapshot_path.exists():
            return None

        try:
            with gzip.open(snapshot_path, "rb") as f:
                snapshot_data = pickle.load(f)

                if isinstance(snapshot_data, dict) and "library_data" in snapshot_data:
                    return snapshot_data
                else:
                    return None
        except Exception:
            return None

    def get_latest_snapshot(self) -> Optional[str]:
        """Get the ID of the latest snapshot"""
        snapshots = list(self.snapshots_path.glob("snapshot_*.pkl.gz"))
        if not snapshots:
            return None

        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = snapshots[0]

        filename = latest.stem.replace(".pkl", "")
        return filename.replace("snapshot_", "")

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots with metadata"""
        snapshots = list(self.snapshots_path.glob("snapshot_*.pkl.gz"))
        snapshot_info = []

        for snapshot_path in snapshots:
            filename = snapshot_path.stem.replace(".pkl", "")
            snapshot_id = filename.split("snapshot_", 1)[1]

            metadata = self.load_snapshot(snapshot_id)
            if metadata:
                snapshot_info.append(
                    {
                        "snapshot_id": snapshot_id,
                        "last_wal_sequence": metadata.get("last_wal_sequence", 0),
                        "created_at": metadata.get("created_at", "unknown"),
                        "file_size": snapshot_path.stat().st_size,
                    }
                )

        return sorted(snapshot_info, key=lambda x: x["created_at"])

    def delete_snapshot(self, snapshot_id: str) -> bool:
        snapshot_path = self.snapshots_path / f"snapshot_{snapshot_id}.pkl.gz"

        if snapshot_path.exists():
            snapshot_path.unlink()
            return True

        return False

    def cleanup_old_snapshots(self, keep_count: int = 1) -> List[int]:
        snapshots = self.list_snapshots()

        deleted_sequences = []

        for snapshot in snapshots[:-keep_count]:
            snapshot_id = snapshot["snapshot_id"]

            metadata = self.load_snapshot(snapshot_id)
            if metadata:
                deleted_sequences.append(metadata.get("last_wal_sequence", 0))

            self.delete_snapshot(snapshot_id)

        return deleted_sequences

    def close(self) -> None:
        self.stop_snapshot_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PersistenceManager:
    def __init__(
        self,
        storage_path: str,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
        snapshot_interval: float = 300.0,
    ):
        self.storage_path = storage_path
        self.wal = BufferedWAL(storage_path, buffer_size, flush_interval)
        self.snapshot_manager = SnapshotManager(storage_path, snapshot_interval)

    def set_snapshot_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        def snapshot_callback():
            library_data = callback()
            last_wal_sequence = self.wal.get_current_sequence()
            return library_data, last_wal_sequence

        self.snapshot_manager.set_snapshot_callback(snapshot_callback)

    def log(self, operation: str, args: Any, kwargs: Any) -> int:
        return self.wal.write_entry(operation, args, kwargs)

    def create_snapshot(
        self, library_data: Dict[str, Any], snapshot_id: Optional[str] = None
    ) -> str:
        self.wal.flush()

        last_wal_sequence = self.wal.get_current_sequence()

        snapshot_id = self.snapshot_manager.create_snapshot(
            library_data, last_wal_sequence, snapshot_id
        )

        return snapshot_id

    def restore_library_state(self) -> Optional[Dict[str, Any]]:
        latest_snapshot_id = self.snapshot_manager.get_latest_snapshot()
        library_data = None
        min_wal_sequence = 0

        if latest_snapshot_id:
            snapshot_data = self.snapshot_manager.load_snapshot(latest_snapshot_id)
            if snapshot_data:
                library_data = snapshot_data.get("library_data")
                min_wal_sequence = snapshot_data.get("last_wal_sequence", 0)

        wal_entries = self.wal.read_entries(min_sequence=min_wal_sequence)

        return {
            "snapshot_data": library_data,
            "wal_entries": wal_entries,
            "last_wal_sequence": min_wal_sequence,
        }

    def flush(self) -> None:
        self.wal.flush()

    def close(self) -> None:
        self.snapshot_manager.close()
        self.wal.close()

    def cleanup(self, keep_snapshots: int = 1) -> None:
        self.snapshot_manager.cleanup_old_snapshots(keep_snapshots)

        remaining_snapshots = self.snapshot_manager.list_snapshots()
        if remaining_snapshots:
            min_sequence_to_keep = min(
                s["last_wal_sequence"] for s in remaining_snapshots
            )
        else:
            min_sequence_to_keep = 0

        if min_sequence_to_keep > 0:
            self.wal.cleanup_old_entries(min_sequence_to_keep)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
