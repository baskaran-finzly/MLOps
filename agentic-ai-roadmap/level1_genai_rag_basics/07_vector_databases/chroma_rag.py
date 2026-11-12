"""Lesson 07 – persistent vector search with Chroma + RAG on top.

Run:  pip install chromadb
      python level1_genai_rag_basics/07_vector_databases/chroma_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb  # pip install chromadb

from common import ask, banner, require_api_key
from common.corpus import build_chunks

DB_DIR = Path(__file__).resolve().parents[2] / ".chroma"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection("mlops_repo", metadata={"hnsw:space": "cosine"})

# Index once; later runs reuse the persisted vectors.
if collection.count() == 0:
    chunks = build_chunks()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )  # Chroma embeds documents with its default model (all-MiniLM-L6-v2)
    print(f"Indexed {len(chunks)} chunks into {DB_DIR}")
else:
    print(f"Reusing {collection.count()} vectors from {DB_DIR}")

banner("Unfiltered semantic query")
q = "how do users interact with the prediction app?"
res = collection.query(query_texts=[q], n_results=3)
for id_, dist in zip(res["ids"][0], res["distances"][0]):
    print(f"  {1 - dist:.2f}  {id_}")

banner("Same query, filtered to deployment/app.py")
res = collection.query(query_texts=[q], n_results=3, where={"source": "deployment/app.py"})
for id_, dist in zip(res["ids"][0], res["distances"][0]):
    print(f"  {1 - dist:.2f}  {id_}")

banner("RAG answer from Chroma results")
require_api_key()
question = "What input fields does the Streamlit app collect from the user?"
res = collection.query(query_texts=[question], n_results=5)
context = "\n".join(
    f'<document id="{i}">\n{doc}\n</document>' for i, doc in zip(res["ids"][0], res["documents"][0])
)
print(ask(
    f"<documents>\n{context}\n</documents>\n\n<question>{question}</question>",
    system="Answer only from the documents and cite document ids in [brackets].",
    max_tokens=2000,
))
