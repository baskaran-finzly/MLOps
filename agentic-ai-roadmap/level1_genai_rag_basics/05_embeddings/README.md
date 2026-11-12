# 05 · Embeddings & Semantic Search

> **Goal:** Understand how text becomes vectors, and how "similar meaning" becomes "nearby vectors".
> (This step is missing from the whiteboard but every RAG system depends on it.)

## Concepts

| Term | Meaning |
|---|---|
| **Embedding** | A list of numbers (e.g. 384 or 1024 floats) representing the *meaning* of a text. |
| **Embedding model** | A model trained so that similar texts get similar vectors (e.g. `all-MiniLM-L6-v2`, Voyage AI, Cohere, OpenAI embeddings). Claude itself generates text, not embeddings — pair it with an embedding model. |
| **Cosine similarity** | Angle between two vectors: 1 = same direction (same meaning), 0 = unrelated. |
| **Sparse vs dense** | Sparse (TF-IDF/BM25) = keyword overlap, great for exact names like `ProdTaken`. Dense = meaning, great for paraphrases ("which column is the target?"). |
| **Hybrid search** | Combine both; usually the best real-world choice. |

```
"How is the model deployed?"   ─► embed ─► [0.12, -0.40, 0.88, …] ┐
"Docker image pushed to HF"    ─► embed ─► [0.10, -0.35, 0.91, …] ┘ cosine ≈ 0.93  ✅ similar
"Age column has missing values"─► embed ─► [-0.70, 0.22, 0.05, …]   cosine ≈ 0.08  ❌ unrelated
```

## Hands-on
```bash
python level1_genai_rag_basics/05_embeddings/embeddings_basics.py            # TF-IDF (offline)
pip install sentence-transformers
python level1_genai_rag_basics/05_embeddings/embeddings_basics.py --dense    # real dense embeddings
```
The script embeds the repo chunks, runs the same queries with sparse and (optionally) dense vectors, and prints the top hits so you can *see* where each wins.

## Exercises
1. Find a query where TF-IDF beats dense embeddings (hint: exact identifiers).
2. Find one where dense wins (hint: synonyms — "ship" vs "deploy").
3. Implement hybrid search: `score = 0.5 * sparse + 0.5 * dense`.
