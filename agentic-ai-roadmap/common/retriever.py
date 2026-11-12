"""A tiny, dependency-light retriever (TF-IDF + cosine similarity).

Lesson 05 explains embeddings; this is the "no extra downloads" version so the
RAG and agent lessons run anywhere scikit-learn is installed. Lesson 07 swaps
it for a real vector database (Chroma).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Chunk, build_chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else build_chunks()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


def format_context(results: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as XML-tagged documents for the prompt."""
    parts = []
    for chunk, score in results:
        parts.append(
            f'<document source="{chunk.metadata["source"]}" chunk="{chunk.metadata["chunk"]}" '
            f'score="{score:.3f}">\n{chunk.text}\n</document>'
        )
    return "\n".join(parts)
