import pytest
from stackdb.models.chunk import Chunk
from stackdb.filter import check_chunk, ChunkFilter, FilterError


class TestFilterBasics:
    @pytest.fixture
    def test_chunks(self):
        return [
            Chunk(
                id="1",
                text="This is about red roses and flowers",
                embedding=[0.1, 0.2, 0.3],
                metadata={"color": "red", "likes": 75, "category": "nature"},
            ),
            Chunk(
                id="2",
                text="Blue ocean waves",
                embedding=[0.4, 0.5, 0.6],
                metadata={"color": "blue", "likes": 42, "category": "nature"},
            ),
            Chunk(
                id="3",
                text="Machine learning algorithms",
                embedding=[0.7, 0.8, 0.9],
                metadata={
                    "color": "green",
                    "likes": 128,
                    "category": "tech",
                    "tags": "AI,ML",
                },
            ),
            Chunk(
                id="4",
                text="Red car driving fast",
                embedding=[0.2, 0.4, 0.6],
                metadata={"color": "red", "likes": 33, "category": "vehicles"},
            ),
        ]

    def test_color_like_filter(self, test_chunks):
        filter_obj = ChunkFilter('color LIKE "red%"')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 2
        assert all(chunk.metadata.get("color") == "red" for chunk in results)
        assert set(chunk.id for chunk in results) == {"1", "4"}

    def test_numeric_comparison(self, test_chunks):
        filter_obj = ChunkFilter("likes > 50")
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 2
        assert all(chunk.metadata.get("likes") > 50 for chunk in results)
        assert set(chunk.id for chunk in results) == {"1", "3"}

    def test_combined_and_filter(self, test_chunks):
        filter_obj = ChunkFilter('color LIKE "red%" AND likes > 50')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].metadata.get("color") == "red"
        assert results[0].metadata.get("likes") > 50

    def test_text_search(self, test_chunks):
        filter_obj = ChunkFilter('text LIKE "%machine%"')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 1
        assert results[0].id == "3"
        assert "machine" in results[0].text.lower()

    def test_metadata_dot_notation(self, test_chunks):
        filter_obj = ChunkFilter('metadata.category = "tech"')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 1
        assert results[0].id == "3"
        assert results[0].metadata.get("category") == "tech"

    def test_or_condition(self, test_chunks):
        filter_obj = ChunkFilter('category = "tech" OR likes > 70')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 2
        result_ids = {chunk.id for chunk in results}
        assert result_ids == {"1", "3"}

    def test_not_condition(self, test_chunks):
        filter_obj = ChunkFilter('NOT color = "blue"')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 3
        assert all(chunk.metadata.get("color") != "blue" for chunk in results)
        assert "2" not in {chunk.id for chunk in results}

    def test_complex_parentheses(self, test_chunks):
        filter_obj = ChunkFilter('(color = "red" OR color = "blue") AND likes >= 40')
        results = filter_obj.filter_chunks(test_chunks)
        assert len(results) == 2
        result_ids = {chunk.id for chunk in results}
        assert result_ids == {"1", "2"}


class TestSingleChunkFilter:
    @pytest.fixture
    def test_chunk(self):
        return Chunk(
            id="test",
            text="Red sports car",
            embedding=[0.1, 0.2, 0.3],
            metadata={"color": "red", "likes": 85, "type": "vehicle"},
        )

    def test_color_like_match(self, test_chunk):
        result = check_chunk(test_chunk, 'color LIKE "red%"')
        assert result is True

    def test_color_like_no_match(self, test_chunk):
        result = check_chunk(test_chunk, 'color LIKE "blue%"')
        assert result is False

    def test_likes_greater_than(self, test_chunk):
        result = check_chunk(test_chunk, "likes > 50")
        assert result is True
        result = check_chunk(test_chunk, "likes > 100")
        assert result is False

    def test_text_search(self, test_chunk):
        result = check_chunk(test_chunk, 'text LIKE "%sports%"')
        assert result is True
        result = check_chunk(test_chunk, 'text LIKE "%boat%"')
        assert result is False

    def test_metadata_dot_notation(self, test_chunk):
        result = check_chunk(test_chunk, 'metadata.type = "vehicle"')
        assert result is True
        result = check_chunk(test_chunk, 'metadata.type = "animal"')
        assert result is False


class TestFilterErrorHandling:
    @pytest.fixture
    def test_chunk(self):
        return Chunk(
            id="test",
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "value"},
        )

    def test_empty_filter(self, test_chunk):
        assert check_chunk(test_chunk, "") is True

    def test_invalid_syntax(self, test_chunk):
        with pytest.raises(FilterError):
            check_chunk(test_chunk, "invalid syntax")

    def test_missing_value(self, test_chunk):
        with pytest.raises(FilterError):
            check_chunk(test_chunk, "field >")

    def test_missing_field(self, test_chunk):
        with pytest.raises(FilterError):
            check_chunk(test_chunk, "= value")

    def test_invalid_operator(self, test_chunk):
        with pytest.raises(FilterError):
            check_chunk(test_chunk, "field INVALID_OP value")


class TestFilterOperators:
    @pytest.fixture
    def numeric_chunk(self):
        return Chunk(
            id="numeric",
            text="Test numeric",
            embedding=[0.1, 0.2, 0.3],
            metadata={"score": 75, "rating": 4.5, "count": 100},
        )

    def test_equality_operators(self, numeric_chunk):
        assert check_chunk(numeric_chunk, "score = 75") is True
        assert check_chunk(numeric_chunk, "score = 50") is False
        assert check_chunk(numeric_chunk, "score != 50") is True
        assert check_chunk(numeric_chunk, "score != 75") is False

    def test_comparison_operators(self, numeric_chunk):
        assert check_chunk(numeric_chunk, "score > 50") is True
        assert check_chunk(numeric_chunk, "score > 100") is False
        assert check_chunk(numeric_chunk, "score < 100") is True
        assert check_chunk(numeric_chunk, "score < 50") is False
        assert check_chunk(numeric_chunk, "score >= 75") is True
        assert check_chunk(numeric_chunk, "score >= 100") is False
        assert check_chunk(numeric_chunk, "score <= 75") is True
        assert check_chunk(numeric_chunk, "score <= 50") is False

    def test_like_operator_wildcards(self):
        chunk = Chunk(
            id="text_test",
            text="Hello World Example",
            embedding=[0.1, 0.2, 0.3],
            metadata={"name": "test_file.txt"},
        )
        assert check_chunk(chunk, 'text LIKE "%World%"') is True
        assert check_chunk(chunk, 'text LIKE "Hello%"') is True
        assert check_chunk(chunk, 'text LIKE "%Example"') is True
        assert check_chunk(chunk, 'text LIKE "%Nonexistent%"') is False
        assert check_chunk(chunk, 'name LIKE "test_file.txt"') is True
        assert check_chunk(chunk, 'name LIKE "test_____.txt"') is True


class TestFilterPerformance:
    def test_large_dataset_filtering(self, large_dataset):
        filter_obj = ChunkFilter("metadata.batch = 5")
        results = filter_obj.filter_chunks(large_dataset)
        assert len(results) == 100
        assert all(chunk.metadata.get("batch") == 5 for chunk in results)

    def test_complex_filter_performance(self, large_dataset):
        filter_obj = ChunkFilter("metadata.index > 500 AND metadata.batch <= 7")
        results = filter_obj.filter_chunks(large_dataset)
        expected_count = 299
        assert len(results) == expected_count
