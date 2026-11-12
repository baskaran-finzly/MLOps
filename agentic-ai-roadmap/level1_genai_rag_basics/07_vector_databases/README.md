# 07 · Vector Databases

> **Goal:** Store and search embeddings at scale with a real vector DB.

Lesson 06 kept vectors in memory. Real systems need persistence, metadata filtering, millions of vectors and fast approximate search.

## Landscape

| DB | Type | Good for |
|---|---|---|
| **Chroma** | Embedded / local, open source | Learning, prototypes, small apps (used here) |
| **FAISS** | Library (Meta) | Fast in-process similarity search, no server |
| **pgvector** | Postgres extension | You already run Postgres; SQL + vectors together |
| **Pinecone** | Managed SaaS | Zero-ops production at scale |
| **Weaviate / Qdrant / Milvus** | Open-source servers (+ cloud) | Self-hosted production, hybrid search, filtering |

## Core operations
`create collection` → `add(ids, documents, embeddings, metadatas)` → `query(text or vector, n_results, where={metadata filter})` → `update` / `delete`.

## Key ideas
- **ANN indexes** (HNSW, IVF) trade a little accuracy for huge speed-ups.
- **Metadata filters** (`where={"source": "deployment/app.py"}`) narrow the search space — essential for multi-tenant apps.
- **Distance metric** must match how the embedding model was trained (usually cosine).
- **Re-index** when you change the embedding model — vectors from different models are not comparable.

## Hands-on
```bash
pip install chromadb
python level1_genai_rag_basics/07_vector_databases/chroma_rag.py
```
Builds a persistent Chroma collection of this repo (`.chroma/`), runs filtered + unfiltered queries, then answers a question with RAG on top of it. (Chroma downloads a small embedding model on first run.)

## Exercises
1. Re-run the script – notice it reuses the persisted index instead of re-embedding.
2. Add a `"kind": "code" | "config" | "docs"` metadata field and filter by it.
3. Swap Chroma for FAISS or pgvector with the same `search()` interface.
