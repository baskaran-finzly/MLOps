"""Lesson 01 – your first GenAI calls.

Run:  python level1_genai_rag_basics/01_genai_intro/hello_claude.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

from common import MODEL, banner, create, require_api_key, text_of

require_api_key()
print(f"Using model: {MODEL}")

# 1) Explain — the raw API call, so you can see its shape.
banner("1) Explain")
response = create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "In 3 bullet points: what is Generative AI and how is it different from classic ML?"}],
)
print(text_of(response))
print(f"\nstop_reason={response.stop_reason}  usage: in={response.usage.input_tokens} out={response.usage.output_tokens}")

# 2) Generate — create something new.
banner("2) Generate")
response = create(
    max_tokens=1024,
    system="You are a senior Python engineer. Reply with code only.",
    messages=[{"role": "user", "content": "Write a Python function that validates an email address with a regex."}],
)
print(text_of(response))

# 3) Transform — rewrite existing content.
banner("3) Transform")
tech = "The XGBoost classifier achieved 0.91 ROC-AUC on the held-out split after Optuna-style grid search."
response = create(
    max_tokens=512,
    messages=[{"role": "user", "content": f"Rewrite this for a non-technical manager in one sentence:\n\n{tech}"}],
)
print(text_of(response))

# What does a response actually contain?
banner("Anatomy of a response")
for block in response.content:
    print(f"- block type: {block.type}")
