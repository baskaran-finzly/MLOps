"""Lesson 03 – prompt engineering patterns, applied to this repo's code.

Run:  python level1_genai_rag_basics/03_prompt_engineering/prompt_patterns.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import REPO_ROOT, ask, banner, create, require_api_key, text_of

require_api_key()
dockerfile = (REPO_ROOT / "deployment" / "Dockerfile").read_text()

# 1) Vague vs specific
banner("1) Vague prompt")
print(ask(f"Explain this:\n{dockerfile}", max_tokens=800))

banner("1) Specific prompt (audience + format + length + purpose)")
print(ask(
    "You are reviewing a teammate's Dockerfile before it ships to Hugging Face Spaces.\n"
    "Give exactly 3 bullets: (1) what the image does, (2) one security or size risk, "
    "(3) one concrete improvement. Max 25 words per bullet.\n\n"
    f"<dockerfile>\n{dockerfile}\n</dockerfile>",
    max_tokens=800,
))

# 2) System prompt as role
banner("2) System prompt")
print(ask(
    "Should I store my HF_TOKEN in the Dockerfile?",
    system="You are a pragmatic DevSecOps mentor. Answer in two sentences, then one code snippet if useful.",
    max_tokens=800,
))

# 3) Few-shot
banner("3) Few-shot classification")
few_shot = """Classify each commit message as feat, fix, docs, or refactor. Reply with the label only.

<example><msg>Add Streamlit UI for predictions</msg><label>feat</label></example>
<example><msg>Fix NaN handling in Age column</msg><label>fix</label></example>
<example><msg>Explain pipeline steps in README</msg><label>docs</label></example>

<msg>{msg}</msg>"""
for msg in ["Refactored train script", "Updated pipeline file with detailed explanation", "Handle missing MonthlyIncome"]:
    print(f"{msg!r:55} -> {ask(few_shot.format(msg=msg), max_tokens=500)}")

# 4) Step-by-step reasoning with a tagged final answer
banner("4) Think, then answer in tags")
out = ask(
    "Our model has precision 0.80 and recall 0.50 on the positive class. "
    "Is F1 above or below 0.6? Reason briefly, then put only 'above' or 'below' in <answer></answer>.",
    max_tokens=2000,
)
print(out)
print("Parsed answer:", out.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in out else "?")

# 5) Structured output enforced by a JSON schema
banner("5) Schema-enforced JSON")
schema = {
    "type": "object",
    "properties": {
        "base_image": {"type": "string"},
        "exposed_port": {"type": "string", "description": "Port number, or 'none'"},
        "entrypoint": {"type": "string"},
        "improvements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["base_image", "exposed_port", "entrypoint", "improvements"],
    "additionalProperties": False,
}
r = create(
    max_tokens=2000,
    output_config={"format": {"type": "json_schema", "schema": schema}},
    messages=[{"role": "user", "content": f"Extract facts from this Dockerfile:\n<dockerfile>\n{dockerfile}\n</dockerfile>"}],
)
print(json.dumps(json.loads(text_of(r)), indent=2))
