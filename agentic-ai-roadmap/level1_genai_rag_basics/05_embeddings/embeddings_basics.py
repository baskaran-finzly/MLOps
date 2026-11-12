"""Lesson 05 – sparse vs dense embeddings over this repo.

Run:  python level1_genai_rag_basics/05_embeddings/embeddings_basics.py [--dense]
No API key needed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.corpus import build_chunks
from common.retriever import TfidfRetriever

parser = argparse.ArgumentParser()
parser.add_argument("--dense", action="store_true", help="also use sentence-transformers dense embeddings")
args = parser.parse_args()

QUERIES = [
    "Which column is the prediction target?",
    "How does the app get shipped to users?",
    "What hyperparameters are tuned?",
    "ProdTaken",
]

chunks = build_chunks()
sparse = TfidfRetriever(chunks)
print(f"Indexed {len(chunks)} chunks. TF-IDF vocabulary size = {len(sparse.vectorizer.vocabulary_):,}")

dense_model = None
if args.dense:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_vecs = dense_model.encode([c.text for c in chunks], normalize_embeddings=True)
    print(f"Dense embedding shape: {chunk_vecs.shape}  (one {chunk_vecs.shape[1]}-dim vector per chunk)")


def preview(text: str) -> str:
    return " ".join(text.split())[:90]


for q in QUERIES:
    print(f"\nQ: {q}")
    for chunk, score in sparse.search(q, k=3):
        print(f"  [sparse {score:.2f}] {chunk.id:38} {preview(chunk.text)}")
    if dense_model is not None:
        qv = dense_model.encode([q], normalize_embeddings=True)[0]
        scores = chunk_vecs @ qv  # cosine similarity (vectors are normalised)
        for i in np.argsort(scores)[::-1][:3]:
            print(f"  [dense  {scores[i]:.2f}] {chunks[i].id:38} {preview(chunks[i].text)}")

# Cosine similarity from scratch – the math behind every vector DB.
a, b = np.array([0.1, 0.9, 0.2]), np.array([0.15, 0.8, 0.3])
print(f"\ncosine([0.1,0.9,0.2],[0.15,0.8,0.3]) = {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.3f}")
