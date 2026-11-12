"""The knowledge base used by the RAG lessons.

Instead of toy data, the RAG lessons answer questions about *this repository*
(the Wellness Tourism MLOps project): its pipeline, training code, and
deployment. That way every answer can be checked against real files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import REPO_ROOT

CORPUS_FILES = [
    "README.md",
    ".github/workflows/pipeline.yml",
    "model_building/data_register.py",
    "model_building/prep.py",
    "model_building/train.py",
    "deployment/app.py",
    "deployment/Dockerfile",
    "deployment/requirements.txt",
    "hosting/hosting.py",
]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(root: Path = REPO_ROOT) -> dict[str, str]:
    """Return {relative_path: file_text} for every corpus file that exists."""
    docs = {}
    for rel in CORPUS_FILES:
        path = root / rel
        if path.exists():
            docs[rel] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def clean_text(text: str) -> str:
    """Light normalisation: strip trailing spaces, collapse 3+ blank lines."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_tail(text: str, n: int) -> str:
    """Last ~n characters of `text`, starting at a line boundary (no half words)."""
    if n <= 0 or not text:
        return ""
    tail = text[-n:]
    cut = tail.find("\n")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else ""


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split on blank lines (paragraph / function boundaries), then pack the
    pieces into chunks of at most `max_chars`, carrying `overlap` characters
    of context from the previous chunk so ideas aren't cut in half."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # A single huge paragraph: hard-split it.
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :]
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = _line_tail(current, overlap)
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks = []
    for source, raw in load_documents().items():
        for i, piece in enumerate(chunk_text(clean_text(raw), max_chars, overlap)):
            chunks.append(Chunk(id=f"{source}#{i}", text=piece, metadata={"source": source, "chunk": i}))
    return chunks
