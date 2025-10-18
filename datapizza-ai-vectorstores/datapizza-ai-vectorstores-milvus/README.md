# Milvus Vectorstore Integration for Datapizza

This module provides a **Milvus vectorstore** implementation compatible with the [Datapizza-AI](https://github.com/datapizza-ai) framework.  
It allows storing, indexing, searching, and managing vector embeddings in a Milvus server, providing both **synchronous** and **asynchronous** interfaces.

---

## Features

- Create and manage collections in Milvus
- Add chunks with dense embeddings
- Create indexes for fast similarity search
- Search for similar vectors
- Remove, retrieve, and dump chunks
- Fully async-compatible for integration with async workflows

---

## Requirements

- Python 3.10+
- [pymilvus](https://milvus.io/docs/install_python.md) (`pip install pymilvus`)
- Datapizza types and core vectorstore

Milvus server must be running locally or remotely.

---

## Installation

Example of docker-compose.yml:
```yml
  etcd:
    # coordinator
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - milvus-net

  minio:
    # for storage upload S3
    image: minio/minio:RELEASE.2024-04-06T05-26-02Z
    container_name: milvus-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./milvus/minio:/data
    networks:
      - milvus-net

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.16
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: minioadmin
      MINIO_SECRET_ACCESS_KEY: minioadmin
      MINIO_USE_SSL: "false"
      MINIO_BUCKET_NAME: milvus-bucket
    command: ["milvus", "run", "standalone"]   # <--- aggiungi questo
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    volumes:
      - ./milvus/db:/var/lib/milvus
    networks:
      - milvus-net



networks:
  milvus-net:
    driver: bridge
```


Docker execution:
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 19121:19121 \
  milvusdb/milvus:2.3.0-standalone
```

Client example for usage:
```python
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding, EmbeddingFormat
from vector_store.milvus_vectorstore import MilvusVectorstore

# initialize MilvusVectorstore
store = MilvusVectorstore(host="127.0.0.1", port="19530", alias="default")

# *- 1. create a collection -*
store.create_collection(
    "docs",
    [VectorConfig(name="vector", dimensions=4, format=EmbeddingFormat.DENSE)]
)

# *- 2. add a chunk -*
chunk = Chunk(
    id="1",
    text="Hello Milvus from Datapizza!",
    metadata={},
    embeddings=[DenseEmbedding(name="vector", vector=[0.1, 0.2, 0.3, 0.4])]
)
store.add(chunk, "docs")

# *- 3. create an index for faster search -*
store.create_index("docs")

# *- 4. load the collection into memory -*
store.load_collection("docs")

# *- 5. search for similar vectors -*
results = store.search("docs", [0.1, 0.2, 0.3, 0.4], k=1)
for r in results:
    print(r.id, r.text, r.embeddings[0].vector)
```

**License**
This project is licensed under the MIT License.