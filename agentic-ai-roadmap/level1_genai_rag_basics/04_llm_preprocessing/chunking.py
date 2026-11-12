"""Lesson 04 – load, clean and chunk this repo's files.

Run:  python level1_genai_rag_basics/04_llm_preprocessing/chunking.py [--contextual]
The default mode is fully offline.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.corpus import build_chunks, chunk_text, clean_text, load_documents

parser = argparse.ArgumentParser()
parser.add_argument("--max-chars", type=int, default=1200)
parser.add_argument("--overlap", type=int, default=200)
parser.add_argument("--contextual", action="store_true", help="add LLM-written context headers (needs API key)")
args = parser.parse_args()

# 1) Load
docs = load_documents()
print(f"Loaded {len(docs)} documents:")
for name, text in docs.items():
    print(f"  {name:40} {len(text):>7,} chars")

# 2) Clean – show the effect on one file
name = "model_building/train.py"
raw = docs[name]
cleaned = clean_text(raw)
print(f"\nCleaning {name}: {len(raw):,} -> {len(cleaned):,} chars")

# 3) Chunk – compare sizes
print("\nChunk-size experiment on the whole corpus:")
for size in (400, args.max_chars, 3000):
    sizes = [len(c) for d in docs.values() for c in chunk_text(clean_text(d), size, args.overlap if size > args.overlap else 0)]
    print(f"  max_chars={size:<5} -> {len(sizes):>4} chunks, median {statistics.median(sizes):>6.0f} chars")

# 4) Metadata – what a chunk record looks like
chunks = build_chunks(args.max_chars, args.overlap)
sample = chunks[len(chunks) // 2]
print(f"\nSample chunk record:\n  id={sample.id}\n  metadata={sample.metadata}\n  text=\n" +
      "\n".join("    | " + line for line in sample.text.splitlines()[:15]))

# 5) Optional: contextual chunking – an LLM writes a 1-line "where am I" header per chunk.
if args.contextual:
    from common import ask, require_api_key

    require_api_key()
    doc_text = docs[sample.metadata["source"]]
    header = ask(
        f"<document>\n{doc_text}\n</document>\n\n"
        f"<chunk>\n{sample.text}\n</chunk>\n\n"
        "Write ONE short sentence situating this chunk within the whole document, to improve search "
        "retrieval of the chunk. Reply with the sentence only.",
        max_tokens=300,
        output_config={"effort": "low"},
    )
    print(f"\nContextual header for {sample.id}:\n  {header}")
    print("\nIn a real pipeline you'd embed `header + '\\n' + chunk.text` instead of the raw chunk.")
