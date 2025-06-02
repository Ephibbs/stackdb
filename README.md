# In-Memory Vector Database

An in-memory vector database built with FastAPI that provides similarity search capabilities using cosine similarity.

## Main Features

- **REST API**: CRUD operations for libraries, documents, and chunks + search written in Python and FastAPI
- **Similarity search**: Cosine similarity vector search with metadata filtering using numpy
- **Index methods**: Index vectors by flat, ivf, or lsh methods
- **Docker support**: Easy containerization and deployment
- **Python Package**: Available as a standalone Python package for embedding in applications

## Additional Features
- **Metadata filtering**: filter metadata using a SQL-like filter syntax. See `stackdb/indexes/utils/filter.py`
- **Persistance to disk**: Persists to disk using 


## Installation

### Python Package Installation

Install StackDB as a Python package:

```bash
# Install from source
git clone https://github.com/ephibbs/stackdb.git
cd stackdb
pip install -e .
```

## Quick Start

### Python Package Usage

```python
import numpy as np
from stackdb import Library, Document, Chunk

# Create a new library
library = Library(
    name="my_library",
    description="My vector database",
    dimension=384  # Embedding dimension
)

# Create a document
doc = Document(
    title="Sample Document",
    content="This is a sample document for testing.",
    metadata={"source": "example"}
)

# Create a chunk with embedding
embedding = np.random.normal(0, 1, 384).tolist()
chunk = Chunk(
    text="This is a sample document for testing.",
    embedding=embedding,
    document_id=doc.id,
    metadata={"type": "text"}
)

# Add to library
library.add_documents([doc])
library.add_chunks([chunk])

# Search for similar vectors
query_vector = np.random.normal(0, 1, 384).tolist()
results = library.search(
    query_vector=query_vector,
    k=5,  # Top 5 results
    filter="type = 'text'"  # Optional metadata filtering
)

# Process results
for chunk, distance in results:
    print(f"Distance: {distance:.4f}")
    print(f"Text: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")
```

### Index Types

StackDB supports multiple index types for different performance characteristics:

```python
from stackdb import Library, FlatIndex, IVFIndex, LSHIndex

# Flat index (brute force, exact results)
library = Library("my_lib", dimension=384, index_type="flat")

# Approximate Methods

# IVF index (inverted file, memory efficient)
library = Library("my_lib", dimension=384, index_type="ivf", index_params={
    "num_clusters": 100,  # Number of clusters for k-means
    "nprobe": 1,  # Number of nearest clusters to search
    "seed": 42
})

# LSH index (locality sensitive hashing, very fast)
library = Library("my_lib", dimension=384, index_type="lsh", index_params={
    "num_tables": 10,  # Number of hash tables
    "hash_size": 10,  # Size of the hash
    "seed": 42
})
```

## API Endpoints

### Health Check

- **GET /** - Health check endpoint
  - Returns API status and total number of libraries

### Library Management

- **POST /libraries** - Create a new library
  - Body: `LibraryCreate` (name, dimension, metadata, index, index_params)
  - Returns: Library ID and success message

- **GET /libraries** - List all libraries
  - Returns: Array of library information with IDs, names, metadata, and document counts

- **GET /libraries/{library_id}** - Get specific library details
  - Returns: Complete library information including document IDs and index information

- **PATCH /libraries/{library_id}** - Update library
  - Body: `LibraryUpdate` (partial update fields)
  - Returns: Success message

- **DELETE /libraries/{library_id}** - Delete library
  - Returns: Success message

### Document Management

- **POST /libraries/{library_id}/documents** - Create documents in a library
  - Body: Array of `DocumentCreate` objects
  - Returns: Success message

- **GET /libraries/{library_id}/documents** - List documents in a library
  - Query params: `document_ids` (optional), `skip` (default: 0), `limit` (default: 100, max: 1000)
  - Returns: Array of document information

- **GET /libraries/{library_id}/documents/{document_id}** - Get specific document
  - Returns: Complete document information including chunk IDs and count

- **PATCH /libraries/{library_id}/documents/{document_id}** - Update document
  - Body: `DocumentUpdate` (partial update fields)
  - Returns: Success message

- **DELETE /libraries/{library_id}/documents/{document_id}** - Delete document
  - Returns: Success message

### Chunk Management

- **POST /libraries/{library_id}/chunks** - Create chunks in a library
  - Body: Array of `Chunk` objects
  - Returns: Success message

- **GET /libraries/{library_id}/chunks** - List chunks in a library
  - Query params: `skip` (default: 0), `limit` (default: 100, max: 1000), `filter` (optional)
  - Returns: Array of chunk objects

- **GET /libraries/{library_id}/chunks/{chunk_id}** - Get specific chunk
  - Returns: Complete chunk object

- **PATCH /libraries/{library_id}/chunks** - Update multiple chunks
  - Body: Array of `ChunkUpdate` objects
  - Returns: Success message

- **DELETE /libraries/{library_id}/chunks/{chunk_id}** - Delete chunks
  - Query param: `chunk_ids` (array of chunk IDs to delete)
  - Returns: Success message

### Vector Search

- **POST /libraries/{library_id}/search** - Perform similarity search
  - Body: `SearchQuery` (query_vector, k, filter, fields)
  - Returns: Array of `SearchResult` objects with distances and chunk data

### REST API

1. **Install API dependencies**:
   ```bash
   pip install -r api/requirements.txt
   ```

2. **Run the application**:
   ```bash
   cd api
   python main.py
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

  ```bash
  docker build -t vector-db .
  docker run -p 8000:8000 vector-db
  ```

### Technical Choices

1. Three main models: Library, Document, Chunk
Each object keeps track of its own information and id. Libraries keep track of documents and docuemtns track chunks both with a dictionary by key id
2. Index class abstraction:
To promote modularity and extensibility, I added a BaseIndex class abstraction that can be subclassed to create variations of indexes for use in libraries.
3. Persistence through write ahead logs and automated snapshots:
Changes are saved to a buffered WAL in memory and synced to disk on an interval or when there are many changes. Snapshots are also saved periodically at a less frequent interval. This allows the db to restart and recover data on failures.
I assign a durable variable a sequence id to link snapshots with WAL logs and recover using both.
4. The API creates CRUD endpoints for each of the models.
5. Synchronous endpoints to use threadpool instead of coroutines which offer little benefit in cpu intensive in memory tasks

### Limitations & Future Work

1. Implement Cosine K Means for IVF index for a precise implementation
2. Create a snapshot of the index itself to avoid rebuilding on startup
3. Create an abstract class for DistanceMetric and use in all Index classes, add euclidean, cosine
4. The WAL could push to a Kafka service for performance and data replication if extended to a cluster
5. Add async to methods that interface with storage for persistence
6. Writes currently block reads and vice versa. Writes are given preference to avoid write starvation. To improve read scalability, implement MVCC.