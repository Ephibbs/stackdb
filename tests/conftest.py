"""
Shared test configuration and fixtures for StackDB tests.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from typing import List, Generator
import sys

# Add project directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from stackdb import Library, Document, Chunk


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample test chunks."""
    return [
        Chunk(
            id="chunk1",
            text="First test chunk about machine learning",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"category": "tech", "likes": 42, "tags": "AI,ML"},
        ),
        Chunk(
            id="chunk2",
            text="Second chunk about nature and forests",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"category": "nature", "likes": 78, "color": "green"},
        ),
        Chunk(
            id="chunk3",
            text="Third chunk discussing red flowers",
            embedding=[0.2, 0.4, 0.6, 0.8],
            metadata={"category": "nature", "likes": 35, "color": "red"},
        ),
        Chunk(
            id="chunk4",
            text="Fourth chunk about blue ocean waves",
            embedding=[0.9, 0.1, 0.3, 0.7],
            metadata={"category": "nature", "likes": 91, "color": "blue"},
        ),
    ]


@pytest.fixture
def sample_documents(sample_chunks) -> List[Document]:
    """Create sample test documents with chunks."""
    doc1 = Document(
        title="Document about technology",
        metadata={"topic": "technology", "author": "test"},
    )
    # Document model doesn't have add_chunk method, so we'll add chunks directly
    doc1.chunks[sample_chunks[0].id] = sample_chunks[0]
    sample_chunks[0].document_id = doc1.id

    doc2 = Document(
        title="Document about nature",
        metadata={"topic": "nature", "author": "test"},
    )
    # Add chunks directly to the chunks dictionary
    for chunk in sample_chunks[1:4]:
        doc2.chunks[chunk.id] = chunk
        chunk.document_id = doc2.id

    return [doc1, doc2]


@pytest.fixture
def in_memory_library() -> Library:
    """Create an in-memory library for testing."""
    return Library(name="Test Library", dimension=4)


@pytest.fixture
def persistent_library(temp_dir) -> Library:
    """Create a persistent library for testing."""
    storage_path = os.path.join(temp_dir, "test_library")
    return Library(
        name="Persistent Test Library", dimension=4, storage_path=storage_path
    )


@pytest.fixture
def populated_library(in_memory_library, sample_documents) -> Library:
    """Create a library populated with test data."""
    in_memory_library.add_documents(sample_documents)
    return in_memory_library


@pytest.fixture
def random_vectors():
    """Generate random test vectors."""

    def _generate(num_vectors: int = 100, dim: int = 128) -> List[List[float]]:
        np.random.seed(42)  # For reproducible tests
        return np.random.randn(num_vectors, dim).astype(np.float32).tolist()

    return _generate


@pytest.fixture
def large_dataset(random_vectors):
    """Create a large dataset for performance testing."""
    vectors = random_vectors(1000, 128)
    chunks = []
    for i, vector in enumerate(vectors):
        chunk = Chunk(
            id=f"perf_chunk_{i}",
            text=f"Performance test chunk {i}",
            embedding=vector,
            metadata={"index": i, "batch": i // 100},
        )
        chunks.append(chunk)
    return chunks


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "slow: Tests that take more than a few seconds")
    config.addinivalue_line(
        "markers", "api: Tests that require API server to be running"
    )


# Pytest collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark API tests
        if "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
