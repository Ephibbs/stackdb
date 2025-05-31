#!/usr/bin/env python3
"""
Basic StackDB Usage Example

This example demonstrates how to use StackDB as a Python package for
vector similarity search.
"""

import numpy as np
from stackdb import Library, Document, Chunk, FlatIndex, LSHIndex, IVFIndex


def main():
    # Create a new library
    library = Library(
        name="example_library",
        dimension=384,  # Common embedding dimension
        index=IVFIndex(dimension=384, num_clusters=10, nprobe=3),
    )

    print(f"Created library: {library.name}")
    print(f"Library ID: {library.id}")
    print(f"Dimension: {library.dimension}")

    # Create some sample documents and chunks
    documents = []
    chunks = []

    # Sample texts and their mock embeddings
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Vector databases enable similarity search",
        "Natural language processing uses neural networks",
    ]

    for i, text in enumerate(texts):
        # Create a document
        doc = Document(
            title=f"Document {i+1}", content=text, metadata={"source": f"example_{i+1}"}
        )
        documents.append(doc)

        # Create a mock embedding (normally you'd use a real embedding model)
        # For this example, we'll use random embeddings
        np.random.seed(i)  # For reproducible results
        embedding = np.random.normal(0, 1, 384)
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.tolist()

        # Create a chunk
        chunk = Chunk(
            text=text,
            embedding=embedding,
            document_id=doc.id,
            metadata={"index": i, "length": len(text)},
        )
        doc.add_chunk(chunk)

    # Add documents and chunks to the library
    library.add_documents(documents)

    print(f"\nAdded {len(documents)} documents to library")
    print(f"Total chunks in library: {len(library.index.chunks)}")

    # Create a query vector (mock embedding for search)
    np.random.seed(0)  # Use same seed as first chunk for similarity
    query_vector = np.random.normal(0, 1, 384)
    query_vector = query_vector / np.linalg.norm(query_vector)
    query_vector = query_vector.tolist()

    # Perform similarity search
    print("\nPerforming similarity search...")
    results = library.search_chunks(
        query=query_vector,
        k=3,  # Get top 3 most similar chunks
        filter=None,  # No metadata filtering
    )

    print(f"\nTop {len(results)} similar chunks:")
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"{i}. Distance: {distance:.4f}")
        print(f"   Text: {chunk.text[:60]}...")
        print(f"   Metadata: {chunk.metadata}")
        print()

    # Example of metadata filtering
    print("Performing filtered search (chunks with length > 50)...")
    filtered_results = library.search_chunks(
        query=query_vector, k=3, filter="metadata.length > 50"  # Filter by text length
    )

    print(f"\nFiltered results ({len(filtered_results)} chunks):")
    for i, (chunk, distance) in enumerate(filtered_results, 1):
        print(f"{i}. Distance: {distance:.4f}")
        print(f"   Text: {chunk.text}")
        print(f"   Length: {chunk.metadata['length']}")
        print()

    # Show library statistics
    print("Library Statistics:")
    print(f"  Library ID: {library.id}")
    print(f"  Library Name: {library.name}")
    print(f"  Documents: {len(library.documents)}")
    print(f"  Chunks: {len(library.index.chunks)}")
    print(f"  Index type: {library.index.get_info()['type']}")
    print(f"  Dimension: {library.dimension}")


if __name__ == "__main__":
    main()
