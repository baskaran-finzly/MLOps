"""Lesson 02 – tokens, context, statelessness, max_tokens and effort.

Run:  python level1_genai_rag_basics/02_llm_basics/tokens_and_context.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import MODEL, REPO_ROOT, banner, client, create, require_api_key, text_of

require_api_key()

# 1) Tokens: count before you send.
banner("1) Counting tokens")
train_py = (REPO_ROOT / "model_building" / "train.py").read_text()
count = client.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": train_py}])
print(f"train.py = {len(train_py):,} characters ≈ {count.input_tokens:,} tokens "
      f"(~{len(train_py) / count.input_tokens:.1f} chars/token)")

# 2) Statelessness: the model only knows what you send in `messages`.
banner("2) The API is stateless – you carry the history")
history = [{"role": "user", "content": "Hi! My name is Priya and I build MLOps pipelines."}]
reply = create(max_tokens=300, messages=history)
history.append({"role": "assistant", "content": reply.content})
history.append({"role": "user", "content": "What's my name and what do I do?"})
print("WITH history   ->", text_of(create(max_tokens=200, messages=history)))
print("WITHOUT history ->", text_of(create(max_tokens=200, messages=[history[-1]])))

# 3) max_tokens truncation.
banner("3) max_tokens truncation")
short = create(max_tokens=40, messages=[{"role": "user", "content": "Explain transformers in detail."}])
print(text_of(short))
print(f"stop_reason = {short.stop_reason}   <- 'max_tokens' means the answer was cut off")

# 4) Effort: how hard the model thinks.
banner("4) Effort levels")
puzzle = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
          "How much does the ball cost? Answer in one line.")
for effort in ("low", "high"):
    r = create(max_tokens=4000, output_config={"effort": effort}, messages=[{"role": "user", "content": puzzle}])
    print(f"effort={effort:<5} out_tokens={r.usage.output_tokens:<5} -> {text_of(r)}")
