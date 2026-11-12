"""Lesson 08 – the same call through different layers.

Run:  python level1_genai_rag_basics/08_api_wrappers/raw_vs_framework.py [--langchain]
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anthropic

from common import MODEL, banner, client, require_api_key

require_api_key()
QUESTION = "In one sentence: what does MLflow do in an ML pipeline?"

# 1) Raw HTTP – everything is manual.
banner("1) Raw HTTP (httpx)")
try:
    import httpx

    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": MODEL, "max_tokens": 1000, "messages": [{"role": "user", "content": QUESTION}]},
            timeout=120,
        )
        r.raise_for_status()
        print(next(b["text"] for b in r.json()["content"] if b["type"] == "text"))
    else:
        print("(skipped – raw HTTP demo needs ANTHROPIC_API_KEY set explicitly)")
except ImportError:
    print("(skipped – pip install httpx)")

# 2) Official SDK – typed objects, retries, timeouts.
banner("2) Official SDK")
t0 = time.time()
msg = client.messages.create(model=MODEL, max_tokens=1000, messages=[{"role": "user", "content": QUESTION}])
print(next(b.text for b in msg.content if b.type == "text"))
print(f"latency={time.time() - t0:.1f}s  tokens in/out={msg.usage.input_tokens}/{msg.usage.output_tokens}  request_id={msg._request_id}")

# 3) SDK streaming – print tokens as they arrive (essential for chat UIs and long outputs).
banner("3) SDK streaming")
with client.messages.stream(
    model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": "List 5 MLOps best practices, one line each."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()

# 4) Error handling – catch specific → general.
banner("4) Error handling")
try:
    client.messages.create(model="claude-does-not-exist", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
except anthropic.NotFoundError as e:
    print(f"NotFoundError (bad model id) – status {e.status_code}")
except anthropic.RateLimitError:
    print("Rate limited – back off and retry")
except anthropic.APIStatusError as e:
    print(f"Other API error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    print("Network problem")

# 5) Optional: LangChain wrapper.
if "--langchain" in sys.argv:
    banner("5) LangChain (langchain-anthropic)")
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_messages(
        [("system", "You are concise."), ("human", "{question}")]
    ) | ChatAnthropic(model=MODEL, max_tokens=1000)
    print(chain.invoke({"question": QUESTION}).content)
