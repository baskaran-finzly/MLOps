# 06 · RAG Fundamentals (Retrieval-Augmented Generation)

> **Goal:** Make an LLM answer from *your* data, with citations, instead of from memory.

## Why RAG?
LLMs don't know your private code/docs and their knowledge has a cutoff. Fine-tuning is slow and expensive and still hallucinates. RAG = **look it up first, then answer**.

```
            ┌──────────── offline (indexing) ────────────┐
 documents ─► clean ─► chunk ─► embed ─► vector store
                                             │
 question ─► embed ─► similarity search ─────┘─► top-k chunks
                                                    │
                    prompt = instructions + chunks + question ─► LLM ─► grounded answer + citations
```

## The 5 things that make RAG good
1. **Chunking quality** (lesson 04) – most RAG failures are retrieval failures.
2. **Retrieval** – hybrid search (keywords + embeddings), `k` around 3-8, optional **re-ranking** with an LLM or cross-encoder.
3. **Prompting** – put documents in XML tags *before* the question; tell the model to answer only from them and to say "I don't know".
4. **Citations** – make the model quote sources so humans can verify.
5. **Evaluation** – measure retrieval hit-rate and answer faithfulness (lesson 18).

## Hands-on
```bash
python level1_genai_rag_basics/06_rag_fundamentals/simple_rag.py
python level1_genai_rag_basics/06_rag_fundamentals/simple_rag.py "What metric is used to evaluate the model?"
```
The script: retrieves chunks of this repo's code (TF-IDF), builds a grounded prompt, and compares **with-RAG vs no-RAG** answers so you see hallucination disappear.

## Exercises
1. Ask something the repo can't answer ("What's the AWS region?") – does it refuse correctly?
2. Add an **LLM re-ranker**: retrieve 10, ask Claude to pick the 3 most relevant chunk ids, answer from those.
3. Add **query rewriting**: ask Claude to rewrite a vague question into 3 search queries; merge the results.
