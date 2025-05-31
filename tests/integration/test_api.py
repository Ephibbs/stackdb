import pytest
import requests
import time


API_BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def api_client():
    """Provides a requests session for API calls."""
    with requests.Session() as session:
        session.headers.update({"Content-Type": "application/json"})
        yield session


@pytest.fixture(scope="session")
def api_health_check(api_client):
    """Ensure the API is running before tests."""
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = api_client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    pytest.fail("API is not running or not accessible")


@pytest.fixture
def test_library(api_client, api_health_check):
    """Create a test library for use in tests."""
    library_data = {
        "name": "test_library",
        "dimension": 4,
        "index": "flat",
        "metadata": {"test": True},
    }

    response = api_client.post(f"{API_BASE_URL}/libraries", json=library_data)
    assert response.status_code == 200
    library_result = response.json()
    library_id = library_result["id"]

    yield library_id

    api_client.delete(f"{API_BASE_URL}/libraries/{library_id}")


@pytest.fixture
def test_document(api_client, test_library):
    """Create a test document for use in chunk tests."""
    document_data = {"title": "Test Document", "metadata": {"test": True}}

    response = api_client.post(
        f"{API_BASE_URL}/libraries/{test_library}/documents", json=[document_data]
    )
    assert response.status_code == 200

    list_response = api_client.get(f"{API_BASE_URL}/libraries/{test_library}/documents")
    assert list_response.status_code == 200
    documents = list_response.json()
    assert len(documents) > 0

    document_id = documents[-1]["id"]
    return document_id


@pytest.fixture
def sample_vectors():
    """Sample vector data for testing."""
    return [
        {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "text": "Vector 1 text",
            "metadata": {"category": "A", "title": "Vector 1"},
        },
        {
            "embedding": [0.5, 0.6, 0.7, 0.8],
            "text": "Vector 2 text",
            "metadata": {"category": "B", "title": "Vector 2"},
        },
        {
            "embedding": [0.9, 1.0, 1.1, 1.2],
            "text": "Vector 3 text",
            "metadata": {"category": "A", "title": "Vector 3"},
        },
    ]


class TestAPIHealth:

    def test_health_check(self, api_client):
        response = api_client.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_api_response_headers(self, api_client):
        response = api_client.get(f"{API_BASE_URL}/")
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]


class TestVectorOperations:

    def test_add_single_chunk(self, api_client, test_library, test_document):
        chunk_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "text": "test_text",
            "metadata": {"test": True, "name": "test_chunk"},
            "document_id": test_document,
        }

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=[chunk_data]
        )
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "id" in result
        assert isinstance(result["id"], str)
        assert len(result["id"]) > 0

    def test_add_multiple_chunks(
        self, api_client, test_library, test_document, sample_vectors
    ):

        for vector in sample_vectors:
            vector["document_id"] = test_document

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=sample_vectors
        )
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "id" in result

        list_response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks"
        )
        assert list_response.status_code == 200
        chunks = list_response.json()
        assert len(chunks) >= len(sample_vectors)

    def test_get_chunk(self, api_client, test_library, test_document):

        chunk_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "text": "get_test_chunk",
            "metadata": {"test": True, "name": "get_test_chunk"},
            "document_id": test_document,
        }

        add_response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=[chunk_data]
        )
        assert add_response.status_code == 200

        list_response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks"
        )
        assert list_response.status_code == 200
        chunks = list_response.json()
        assert len(chunks) > 0

        chunk_id = chunks[0]["id"]
        get_response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks/{chunk_id}"
        )
        assert get_response.status_code == 200

        chunk = get_response.json()
        assert chunk["id"] == chunk_id
        assert "embedding" in chunk
        assert "metadata" in chunk

    def test_get_nonexistent_chunk(self, api_client, api_health_check, test_library):
        response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks/nonexistent_id"
        )
        assert response.status_code == 404

    def test_list_chunks(
        self, api_client, api_health_check, test_library, test_document, sample_vectors
    ):

        for vector in sample_vectors:
            vector["document_id"] = test_document

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=sample_vectors
        )
        assert response.status_code == 200

        response = api_client.get(f"{API_BASE_URL}/libraries/{test_library}/chunks")
        assert response.status_code == 200

        chunks = response.json()
        assert isinstance(chunks, list)
        assert len(chunks) >= len(sample_vectors)

        for chunk in chunks:
            assert "id" in chunk
            assert "embedding" in chunk
            assert "metadata" in chunk

    def test_update_chunk(
        self, api_client, api_health_check, test_library, test_document
    ):

        original_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "text": "original_chunk",
            "metadata": {"test": True, "name": "original_chunk"},
            "document_id": test_document,
        }

        add_response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=[original_data]
        )
        assert add_response.status_code == 200

        list_response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks"
        )
        chunks = list_response.json()
        chunk_id = chunks[0]["id"]

        updated_data = [
            {
                "id": chunk_id,
                "text": "updated_chunk",
                "metadata": {"test": True, "name": "updated_chunk"},
            }
        ]

        update_response = api_client.patch(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=updated_data
        )
        assert update_response.status_code == 200

        get_response = api_client.get(
            f"{API_BASE_URL}/libraries/{test_library}/chunks/{chunk_id}"
        )
        updated_chunk = get_response.json()
        assert updated_chunk["metadata"]["name"] == "updated_chunk"


class TestSearchOperations:

    @pytest.fixture(autouse=True)
    def setup_search_data(
        self, api_client, test_library, test_document, sample_vectors
    ):
        """Add sample vectors to the library for search tests."""

        for vector in sample_vectors:
            vector["document_id"] = test_document

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=sample_vectors
        )
        assert response.status_code == 200

    def test_basic_search(self, api_client, test_library):
        search_query = {
            "query": [0.1, 0.2, 0.3, 0.4],
            "k": 3,
        }

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/search", json=search_query
        )
        assert response.status_code == 200

        results = response.json()
        assert isinstance(results, list)
        assert len(results) <= 3

        for result in results:
            assert "distance" in result
            assert "chunk" in result
            assert isinstance(result["distance"], (int, float))
            assert "id" in result["chunk"]
            assert "embedding" in result["chunk"]
            assert "metadata" in result["chunk"]

    def test_search_with_high_threshold(self, api_client, test_library):
        search_query = {
            "query": [999.0, 999.0, 999.0, 999.0],
            "k": 5,
        }

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/search", json=search_query
        )
        print("response", response.json())
        assert response.status_code == 200

        results = response.json()
        assert isinstance(results, list)

    def test_search_top_k_limit(self, api_client, test_library):
        search_query = {
            "query": [0.1, 0.2, 0.3, 0.4],
            "k": 1,
        }

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/search", json=search_query
        )
        assert response.status_code == 200

        results = response.json()
        assert len(results) == 1

    def test_search_invalid_vector_dimension(self, api_client, test_library):
        search_query = {
            "query": [0.1, 0.2],
            "k": 3,
        }

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/search", json=search_query
        )
        assert response.status_code == 400


class TestErrorHandling:

    def test_invalid_json_request(self, api_client, test_library):
        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, api_client, test_library):
        incomplete_data = [{"metadata": {"test": True}}]
        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=incomplete_data
        )
        assert response.status_code == 422

    def test_invalid_endpoint(self, api_client):
        response = api_client.get(f"{API_BASE_URL}/nonexistent_endpoint")
        assert response.status_code == 404


@pytest.mark.slow
class TestAPIPerformance:

    def test_bulk_chunk_addition_performance(
        self, api_client, api_health_check, test_library, test_document
    ):
        """Test performance of adding many chunks at once."""
        num_chunks = 100
        bulk_data = []

        for i in range(num_chunks):
            chunk_data = {
                "embedding": [i * 0.01, i * 0.02, i * 0.03, i * 0.04],
                "text": f"test_text_{i}",
                "metadata": {"index": i, "test": "performance"},
                "document_id": test_document,
            }
            bulk_data.append(chunk_data)

        start_time = time.time()
        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=bulk_data
        )
        end_time = time.time()

        assert response.status_code == 200
        duration = end_time - start_time

        assert duration < 10.0

        print(f"Added {num_chunks} chunks in {duration:.2f} seconds")

    def test_search_performance(self, api_client, test_library, test_document):
        """Test search performance after adding many chunks."""

        num_chunks = 50
        bulk_data = []

        for i in range(num_chunks):
            chunk_data = {
                "embedding": [i * 0.01, i * 0.02, i * 0.03, i * 0.04],
                "text": f"test_text_{i}",
                "metadata": {"index": i, "test": "search_performance"},
                "document_id": test_document,
            }
            bulk_data.append(chunk_data)

        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/chunks", json=bulk_data
        )
        print("status code", response.status_code)
        print(response.json())
        assert response.status_code == 200

        search_query = {
            "query": [0.25, 0.5, 0.75, 1.0],
            "k": 10,
        }

        start_time = time.time()
        response = api_client.post(
            f"{API_BASE_URL}/libraries/{test_library}/search", json=search_query
        )
        end_time = time.time()

        assert response.status_code == 200
        duration = end_time - start_time

        assert duration < 2.0

        results = response.json()
        assert len(results) <= 10

        print(f"Search over {num_chunks} chunks completed in {duration:.3f} seconds")
