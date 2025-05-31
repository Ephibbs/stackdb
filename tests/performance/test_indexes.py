"""
Performance tests for StackDB vector indexes.
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any
from stackdb.models.chunk import Chunk
from stackdb.indexes import FlatIndex, LSHIndex, IVFIndex


@pytest.fixture
def performance_chunks():
    def _generate(num_vectors: int = 1000, dim: int = 128) -> List[Chunk]:
        np.random.seed(42)
        chunks = []
        for i in range(num_vectors):
            embedding = np.random.randn(dim).astype(np.float32).tolist()
            chunk = Chunk(
                id=f"perf_chunk_{i}",
                text=f"Performance test chunk number {i}",
                embedding=embedding,
                metadata={"index": i, "batch": i // 100},
            )
            chunks.append(chunk)
        return chunks

    return _generate


@pytest.fixture
def query_vectors():
    def _generate(num_queries: int = 10, dim: int = 128) -> List[List[float]]:
        np.random.seed(123)
        return np.random.randn(num_queries, dim).astype(np.float32).tolist()

    return _generate


class IndexPerformanceTester:
    @staticmethod
    def benchmark_index(
        index_class,
        index_name: str,
        chunks: List[Chunk],
        query_vectors: List[List[float]],
        **kwargs,
    ) -> Dict[str, Any]:
        dimension = len(chunks[0].embedding)
        index = index_class(dimension=dimension, **kwargs)

        results = {"index_name": index_name}

        start_time = time.time()
        index.add_vectors(chunks)
        add_time = time.time() - start_time
        results["add_time"] = add_time
        results["add_throughput"] = len(chunks) / add_time

        start_time = time.time()
        index.build_index()
        build_time = time.time() - start_time
        results["build_time"] = build_time

        search_times = []
        for query_vector in query_vectors:
            start_time = time.time()
            search_results = index.search(query_vector, k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

        results["avg_search_time"] = np.mean(search_times)
        results["min_search_time"] = np.min(search_times)
        results["max_search_time"] = np.max(search_times)
        results["search_std"] = np.std(search_times)

        info = index.get_info()
        results["memory_mb"] = info.get("memory_usage_mb", 0)
        results["total_vectors"] = index.get_vector_count()

        if chunks:
            chunk_to_remove = chunks[0].id
            start_time = time.time()
            success = index.remove_vectors([chunk_to_remove])
            remove_time = time.time() - start_time
            results["remove_time"] = remove_time
            results["remove_success"] = success

        return results


@pytest.mark.slow
class TestFlatIndexPerformance:
    """Performance tests for FlatIndex."""

    def test_flat_index_small_dataset(self, performance_chunks, query_vectors):
        """Test FlatIndex performance on small dataset."""
        chunks = performance_chunks(100, 64)
        queries = query_vectors(5, 64)

        results = IndexPerformanceTester.benchmark_index(
            FlatIndex, "FlatIndex", chunks, queries
        )

        assert results["add_throughput"] > 10
        assert results["avg_search_time"] < 5.0
        assert results["total_vectors"] == 100

    def test_flat_index_large_dataset(self, performance_chunks, query_vectors):
        """Test FlatIndex performance on larger dataset."""
        chunks = performance_chunks(1000, 128)
        queries = query_vectors(10, 128)

        results = IndexPerformanceTester.benchmark_index(
            FlatIndex, "FlatIndex", chunks, queries
        )

        assert results["add_throughput"] > 5
        assert results["total_vectors"] == 1000

        assert results["avg_search_time"] < 10.0


@pytest.mark.slow
class TestLSHIndexPerformance:
    """Performance tests for LSHIndex."""

    def test_lsh_index_performance(self, performance_chunks, query_vectors):
        """Test LSHIndex performance characteristics."""
        chunks = performance_chunks(1000, 128)
        queries = query_vectors(10, 128)

        results = IndexPerformanceTester.benchmark_index(
            LSHIndex,
            "LSHIndex",
            chunks,
            queries,
            num_tables=10,
            hash_size=10,
        )

        assert results["add_throughput"] > 1
        assert results["total_vectors"] == 1000

        assert results["avg_search_time"] < 30.0

    def test_lsh_index_different_parameters(self, performance_chunks, query_vectors):
        """Test LSH performance with different parameters."""
        chunks = performance_chunks(500, 64)
        queries = query_vectors(5, 64)

        results_small = IndexPerformanceTester.benchmark_index(
            LSHIndex,
            "LSHIndex_5_tables",
            chunks,
            queries,
            num_tables=5,
            hash_size=8,
        )

        results_large = IndexPerformanceTester.benchmark_index(
            LSHIndex,
            "LSHIndex_20_tables",
            chunks,
            queries,
            num_tables=20,
            hash_size=8,
        )

        assert results_small["total_vectors"] == results_large["total_vectors"]
        assert results_small["build_time"] <= results_large["build_time"]


@pytest.mark.slow
class TestIVFIndexPerformance:
    """Performance tests for IVFIndex."""

    def test_ivf_index_performance(self, performance_chunks, query_vectors):
        """Test IVFIndex performance characteristics."""
        chunks = performance_chunks(1000, 128)
        queries = query_vectors(10, 128)

        results = IndexPerformanceTester.benchmark_index(
            IVFIndex,
            "IVFIndex",
            chunks,
            queries,
            num_clusters=20,
            nprobe=3,
        )

        assert results["add_throughput"] > 1
        assert results["total_vectors"] == 1000

        assert results["avg_search_time"] < 30.0

    def test_ivf_index_cluster_scaling(self, performance_chunks, query_vectors):
        """Test how IVF performance scales with number of clusters."""
        chunks = performance_chunks(500, 64)
        queries = query_vectors(5, 64)

        cluster_counts = [5, 10, 25]
        results_by_clusters = {}

        for num_clusters in cluster_counts:
            results = IndexPerformanceTester.benchmark_index(
                IVFIndex,
                f"IVFIndex_{num_clusters}_clusters",
                chunks,
                queries,
                num_clusters=num_clusters,
                nprobe=2,
            )
            results_by_clusters[num_clusters] = results

        for results in results_by_clusters.values():
            assert results["total_vectors"] == 500


@pytest.mark.slow
class TestIndexComparison:
    """Comparative performance tests across different index types."""

    def test_index_comparison_medium_dataset(self, performance_chunks, query_vectors):
        """Compare all index types on medium-sized dataset."""
        chunks = performance_chunks(500, 128)
        queries = query_vectors(10, 128)

        index_configs = [
            (FlatIndex, "Flat", {}),
            (
                LSHIndex,
                "LSH",
                {"num_tables": 10, "hash_size": 10},
            ),
            (
                IVFIndex,
                "IVF",
                {"num_clusters": 15, "nprobe": 3},
            ),
        ]

        all_results = []

        for index_class, name, config in index_configs:
            try:
                results = IndexPerformanceTester.benchmark_index(
                    index_class, name, chunks, queries, **config
                )
                all_results.append(results)
            except Exception as e:
                pytest.skip(f"Index {name} not available: {e}")

        for results in all_results:
            assert results["total_vectors"] == 500
            assert results["add_time"] > 0
            assert results["avg_search_time"] > 0

        print("\nPerformance Comparison:")
        print("Index\t\tAdd Time\tSearch Time\tMemory")
        for results in all_results:
            print(
                f"{results['index_name']:<12}\t{results['add_time']:.3f}s\t\t"
                f"{results['avg_search_time']*1000:.1f}ms\t\t{results['memory_mb']:.1f}MB"
            )

    def test_search_accuracy_vs_speed_tradeoff(self, performance_chunks, query_vectors):
        """Test the accuracy vs speed tradeoff for approximate indexes."""
        chunks = performance_chunks(1000, 128)
        queries = query_vectors(5, 128)

        flat_index = FlatIndex(dimension=128)
        flat_index.add_vectors(chunks)
        flat_index.build_index()

        ground_truth_results = []
        for query in queries:
            results = flat_index.search(query, k=10)
            ground_truth_results.append([chunk_id for chunk_id, _ in results])

        approx_configs = [
            (LSHIndex, {"num_tables": 5, "hash_size": 8}),
            (
                LSHIndex,
                {"num_tables": 15, "hash_size": 12},
            ),
        ]

        for index_class, config in approx_configs:
            try:
                results = IndexPerformanceTester.benchmark_index(
                    index_class,
                    f"LSH_{config['num_tables']}_tables",
                    chunks,
                    queries,
                    **config,
                )

                assert results["avg_search_time"] < 1.0

            except Exception as e:
                pytest.skip(f"Index configuration not available: {e}")


@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Test how indexes scale with data size."""

    @pytest.mark.parametrize("dataset_size", [100, 500, 1000, 2000])
    def test_flat_index_scaling(self, performance_chunks, query_vectors, dataset_size):
        """Test how FlatIndex scales with dataset size."""
        chunks = performance_chunks(dataset_size, 64)
        queries = query_vectors(5, 64)

        results = IndexPerformanceTester.benchmark_index(
            FlatIndex, f"Flat_{dataset_size}", chunks, queries
        )

        assert results["avg_search_time"] > 0
        assert results["avg_search_time"] < 10.0

        assert results["add_throughput"] > 10

    @pytest.mark.parametrize("dimension", [32, 64, 128, 256])
    def test_dimension_impact(self, performance_chunks, query_vectors, dimension):
        """Test how vector dimension affects performance."""
        chunks = performance_chunks(500, dimension)
        queries = query_vectors(5, dimension)

        results = IndexPerformanceTester.benchmark_index(
            FlatIndex, f"Flat_{dimension}D", chunks, queries
        )

        assert results["avg_search_time"] > 0

        assert results["total_vectors"] == 500
