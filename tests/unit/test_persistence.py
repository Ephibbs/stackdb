"""
Unit tests for StackDB persistence functionality.
"""

import pytest
import os
from stackdb import Library, Document, Chunk


class TestPersistenceBasics:
    def test_in_memory_library_not_persistent(self, in_memory_library):
        assert in_memory_library.persistence_manager is None

    def test_persistent_library_has_persistence_manager(self, persistent_library):
        assert persistent_library.persistence_manager is not None

    def test_in_memory_library_storage_path(self, in_memory_library):
        assert in_memory_library.storage_path is None

    def test_persistent_library_storage_path(self, persistent_library):
        assert persistent_library.storage_path is not None


class TestWALOperations:

    def test_wal_sequence_increments(self, persistent_library, sample_documents):
        initial_sequence = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        persistent_library.add_documents([sample_documents[0]])

        after_add_sequence = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        assert after_add_sequence > initial_sequence

    def test_flush_persistence(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        persistent_library.flush()

        current_sequence = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )
        assert current_sequence >= 0

    def test_multiple_operations_wal_sequence(self, persistent_library):
        initial_sequence = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        for i in range(3):
            doc = Document(id=f"doc_{i}", title=f"Document {i}")
            doc.chunks[f"chunk_{i}"] = Chunk(
                id=f"chunk_{i}",
                text=f"Chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            )
            persistent_library.add_documents([doc])

        final_sequence = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        assert final_sequence >= initial_sequence + 3


class TestSnapshotOperations:

    def test_create_snapshot(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        snapshot_id = persistent_library.create_snapshot()
        assert snapshot_id is not None
        assert isinstance(snapshot_id, str)
        assert len(snapshot_id) > 0

    def test_create_snapshot_with_name(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        custom_name = "test_snapshot"
        snapshot_id = persistent_library.create_snapshot(custom_name)
        assert snapshot_id == custom_name

    def test_list_snapshots_empty(self, persistent_library):
        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert isinstance(snapshots, list)
        assert len(snapshots) == 0

    def test_list_snapshots_with_data(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        snapshot1_id = persistent_library.create_snapshot("snapshot1")
        snapshot2_id = persistent_library.create_snapshot("snapshot2")

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots) == 2

        snapshot_ids = [s["snapshot_id"] for s in snapshots]
        assert snapshot1_id in snapshot_ids
        assert snapshot2_id in snapshot_ids

        for snapshot in snapshots:
            assert "snapshot_id" in snapshot
            assert "last_wal_sequence" in snapshot
            assert "created_at" in snapshot
            assert "file_size" in snapshot

    def test_snapshot_captures_wal_sequence(self, persistent_library, sample_documents):

        persistent_library.add_documents([sample_documents[0]])

        wal_seq_before = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        snapshot_id = persistent_library.create_snapshot()

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        snapshot = next(s for s in snapshots if s["snapshot_id"] == snapshot_id)

        assert snapshot["last_wal_sequence"] <= wal_seq_before


class TestSnapshotCleanup:

    def test_cleanup_snapshots(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        snapshot_ids = []
        for i in range(5):

            doc = Document(id=f"cleanup_doc_{i}", title=f"Cleanup doc {i}")
            doc.chunks[f"cleanup_chunk_{i}"] = Chunk(
                id=f"cleanup_chunk_{i}",
                text=f"Cleanup chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            )
            persistent_library.add_documents([doc])

            snapshot_id = persistent_library.create_snapshot(f"cleanup_snapshot_{i}")
            snapshot_ids.append(snapshot_id)

        snapshots_before = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots_before) == 5

        persistent_library.persistence_manager.snapshot_manager.cleanup_old_snapshots(
            keep_count=2
        )

        snapshots_after = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots_after) == 2

        remaining_ids = [s["snapshot_id"] for s in snapshots_after]
        assert "cleanup_snapshot_3" in remaining_ids
        assert "cleanup_snapshot_4" in remaining_ids

    def test_cleanup_no_snapshots(self, persistent_library):
        """Test cleanup when no snapshots exist."""

        persistent_library.persistence_manager.snapshot_manager.cleanup_old_snapshots(
            keep_count=2
        )

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots) == 0

    def test_cleanup_fewer_than_keep_count(self, persistent_library, sample_documents):
        """Test cleanup when fewer snapshots exist than keep_count."""
        persistent_library.add_documents(sample_documents)

        persistent_library.create_snapshot("snap1")
        persistent_library.create_snapshot("snap2")

        persistent_library.persistence_manager.snapshot_manager.cleanup_old_snapshots(
            keep_count=5
        )

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots) == 2


class TestPersistenceLifecycle:

    def test_library_restore_from_persistence(self, temp_dir, sample_documents):
        storage_path = os.path.join(temp_dir, "lifecycle_test")

        original_library = Library(
            name="Lifecycle Test", dimension=4, storage_path=storage_path
        )

        original_library.add_documents(sample_documents)
        original_doc_count = len(original_library.documents)

        snapshot_id = original_library.create_snapshot()

        original_library.flush()

        original_library.persistence_manager.close()
        original_library.persistence_manager = None

        restored_library = Library.restore_from_persistence(
            id=original_library.id, storage_path=storage_path
        )

        assert len(restored_library.documents) == original_doc_count

        for doc_id in sample_documents[0].id, sample_documents[1].id:
            assert doc_id in restored_library.documents

        snapshots = (
            restored_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots) >= 1
        assert any(s["snapshot_id"] == snapshot_id for s in snapshots)

        restored_library.persistence_manager.close()

    def test_persistence_disabled_operations(self, in_memory_library, sample_documents):
        in_memory_library.add_documents(sample_documents)

        snapshot_id = in_memory_library.create_snapshot()
        assert snapshot_id is None

        in_memory_library.flush()

    def test_persistence_manager_cleanup(self, persistent_library, sample_documents):
        persistent_library.add_documents(sample_documents)

        print(persistent_library.persistence_manager.snapshot_manager.list_snapshots())

        persistent_library.create_snapshot("snap1")
        persistent_library.create_snapshot("snap2")
        persistent_library.create_snapshot("snap3")

        print(persistent_library.persistence_manager.snapshot_manager.list_snapshots())

        persistent_library.persistence_manager.snapshot_manager.cleanup_old_snapshots(
            keep_count=2
        )

        print(persistent_library.persistence_manager.snapshot_manager.list_snapshots())

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        assert len(snapshots) == 2

    def test_wal_and_snapshot_integration(self, persistent_library, sample_documents):
        persistent_library.add_documents([sample_documents[0]])

        seq_after_first = (
            persistent_library.persistence_manager.wal.get_current_sequence()
        )

        snapshot_id = persistent_library.create_snapshot()

        persistent_library.add_documents([sample_documents[1]])

        final_seq = persistent_library.persistence_manager.wal.get_current_sequence()

        assert final_seq > seq_after_first

        snapshots = (
            persistent_library.persistence_manager.snapshot_manager.list_snapshots()
        )
        snapshot = next(s for s in snapshots if s["snapshot_id"] == snapshot_id)
        assert snapshot["last_wal_sequence"] >= seq_after_first
