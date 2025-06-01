import pytest
from stackdb import Library, Document, Chunk, DocumentUpdate
from pydantic import ValidationError


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(
            text="Test chunk text",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"category": "test"},
        )
        assert chunk.id is not None
        assert chunk.text == "Test chunk text"
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]
        assert chunk.metadata == {"category": "test"}
        assert chunk.created_at is not None

    def test_chunk_with_custom_id(self):
        chunk = Chunk(
            id="custom_id", text="Test text", embedding=[0.1, 0.2], document_id="doc1"
        )
        assert chunk.id == "custom_id"
        assert chunk.document_id == "doc1"

    def test_chunk_validation_errors(self):
        with pytest.raises(ValidationError):
            Chunk(text="", embedding=[0.1])
        with pytest.raises(ValidationError):
            Chunk(text="test", embedding=[])

    def test_chunk_update(self):
        chunk = Chunk(
            text="Original", embedding=[0.1, 0.2], metadata={"existing": "value"}
        )
        chunk.update(text="Updated text", metadata={"new": "value"}, document_id="doc1")
        assert chunk.text == "Updated text"
        assert chunk.metadata == {"existing": "value", "new": "value"}
        assert chunk.document_id == "doc1"


class TestDocument:
    def test_document_creation(self):
        doc = Document(title="Test Document", metadata={"author": "test"})
        assert doc.id is not None
        assert doc.title == "Test Document"
        assert doc.metadata == {"author": "test"}
        assert doc.chunks == {}
        assert doc.created_at is not None

    def test_document_with_custom_id(self):
        doc = Document(id="custom_doc_id", title="Custom Doc", library_id="lib1")
        assert doc.id == "custom_doc_id"
        assert doc.library_id == "lib1"

    def test_document_validation_errors(self):
        with pytest.raises(ValidationError):
            Document(title="")

    def test_document_chunks_direct_access(self):
        doc = Document(title="Test Doc")
        chunk = Chunk(id="chunk1", text="Test chunk", embedding=[0.1, 0.2])
        doc.chunks[chunk.id] = chunk
        assert "chunk1" in doc.chunks
        assert doc.chunks["chunk1"] == chunk
        assert len(doc.chunks) == 1
        del doc.chunks[chunk.id]
        assert len(doc.chunks) == 0

    def test_document_multiple_chunks(self):
        doc = Document(title="Test Doc")
        chunk1 = Chunk(id="c1", text="Text 1", embedding=[0.1])
        chunk2 = Chunk(id="c2", text="Text 2", embedding=[0.2])
        doc.chunks[chunk1.id] = chunk1
        doc.chunks[chunk2.id] = chunk2
        chunk_list = list(doc.chunks.values())
        assert len(chunk_list) == 2
        assert chunk1 in chunk_list
        assert chunk2 in chunk_list

    def test_document_update(self):
        doc = Document(title="Original Title", metadata={"existing": "value"})
        doc.update(title="Updated Title", metadata={"updated": True}, library_id="lib1")
        assert doc.title == "Updated Title"
        assert doc.metadata == {"existing": "value", "updated": True}
        assert doc.library_id == "lib1"


class TestLibrary:
    def test_library_creation(self):
        library = Library(name="Test Library", dimension=4, metadata={"version": "1.0"})
        assert library.id is not None
        assert library.name == "Test Library"
        assert library.dimension == 4
        assert library.metadata == {"version": "1.0"}
        assert library.documents == {}
        assert library.index is not None
        assert library.created_at is not None

    def test_library_validation_errors(self):
        with pytest.raises(ValidationError):
            Library(name="", dimension=4)
        with pytest.raises(ValidationError):
            Library(name="Test", dimension=0)

    def test_library_add_documents(self, sample_documents):
        library = Library(name="Test", dimension=4)
        library.add_documents(sample_documents)
        assert len(library.documents) == 2
        assert sample_documents[0].id in library.documents
        assert sample_documents[1].id in library.documents

    def test_library_remove_documents(self, sample_documents):
        library = Library(name="Test", dimension=4)
        library.add_documents(sample_documents)
        library.remove_documents([sample_documents[0].id])
        assert len(library.documents) == 1
        assert sample_documents[0].id not in library.documents
        assert sample_documents[1].id in library.documents

    def test_library_get_documents(self, sample_documents):
        library = Library(name="Test", dimension=4)
        library.add_documents(sample_documents)
        all_docs = library.get_documents()
        assert len(all_docs) == 2
        specific_docs = library.get_documents([sample_documents[0].id])
        assert len(specific_docs) == 1
        assert specific_docs[0].id == sample_documents[0].id

    def test_library_add_chunks(self, sample_chunks):
        library = Library(name="Test", dimension=4)
        doc = Document(title="Test Doc")
        library.add_documents([doc])
        for chunk in sample_chunks:
            chunk.document_id = doc.id
        library.add_chunks(sample_chunks)
        assert len(doc.chunks) == len(sample_chunks)
        all_chunks = library.get_chunks()
        assert len(all_chunks) == len(sample_chunks)

    def test_library_search_chunks(self, populated_library):
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = populated_library.search_chunks(query_vector, k=2)
        assert len(results) <= 2
        for chunk, distance in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(distance, (int, float))

    def test_library_get_chunks(self, populated_library):
        chunks = populated_library.get_chunks()
        assert len(chunks) > 0
        filtered_chunks = populated_library.get_chunks(
            filter="metadata.category = 'tech'"
        )
        assert all(
            chunk.metadata.get("category") == "tech" for chunk in filtered_chunks
        )

    def test_library_update_document(self, sample_documents):
        library = Library(name="Test", dimension=4)
        library.add_documents(sample_documents)
        doc_id = sample_documents[0].id
        library.update_document(
            DocumentUpdate(id=doc_id, title="Updated Title", metadata={"updated": True})
        )
        updated_doc = library.documents[doc_id]
        assert updated_doc.title == "Updated Title"
        assert "updated" in updated_doc.metadata
        assert updated_doc.metadata["updated"] == True


class TestIntegration:
    def test_full_workflow(self):
        library = Library(name="Integration Test", dimension=3)
        doc = Document(title="Test Document")
        chunk1 = Chunk(
            text="First chunk", embedding=[0.1, 0.2, 0.3], metadata={"type": "intro"}
        )
        chunk2 = Chunk(
            text="Second chunk", embedding=[0.4, 0.5, 0.6], metadata={"type": "body"}
        )
        doc.chunks[chunk1.id] = chunk1
        doc.chunks[chunk2.id] = chunk2
        chunk1.document_id = doc.id
        chunk2.document_id = doc.id
        library.add_documents([doc])
        assert len(library.documents) == 1
        assert len(library.get_chunks()) == 2
        results = library.search_chunks([0.1, 0.2, 0.3], k=1)
        assert len(results) == 1
        library.remove_documents([doc.id])
        assert len(library.documents) == 0
        assert len(library.get_chunks()) == 0
