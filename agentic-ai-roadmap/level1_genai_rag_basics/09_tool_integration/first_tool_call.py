"""Lesson 09 – one tool-use round trip, done by hand so every step is visible.

Run:  python level1_genai_rag_basics/09_tool_integration/first_tool_call.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import banner, create, require_api_key, text_of
from common.tools import calculator, dataset_stats, execute, schema

require_api_key()

tools = [schema(dataset_stats), schema(calculator)]
banner("Tool definitions sent to the model")
print(json.dumps(tools[0], indent=2))

question = "What's the average MonthlyIncome in our customer dataset, and what would a 12% raise make it?"
messages = [{"role": "user", "content": question}]

# Step 1: ask – the model may answer or request tools.
banner(f"Step 1 – user asks: {question}")
response = create(max_tokens=4000, tools=tools, messages=messages)
print(f"stop_reason = {response.stop_reason}")
for block in response.content:
    if block.type == "text" and block.text.strip():
        print(f"  text: {block.text.strip()}")
    elif block.type == "tool_use":
        print(f"  tool_use: {block.name}({json.dumps(block.input)})  id={block.id}")

# Steps 2-3: run each requested tool and return ALL results in ONE user message.
# (This lesson does a single round; if the model needs more, lesson 11 adds the loop.)
rounds = 0
while response.stop_reason == "tool_use" and rounds < 3:
    rounds += 1
    messages.append({"role": "assistant", "content": response.content})  # keep the model's turn verbatim
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output, is_error = execute(block.name, block.input)
            print(f"  ↳ ran {block.name} -> {output[:200]}{' (ERROR)' if is_error else ''}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
    messages.append({"role": "user", "content": results})

    banner(f"Round {rounds} – send tool results back")
    response = create(max_tokens=4000, tools=tools, messages=messages)
    print(f"stop_reason = {response.stop_reason}")
    for block in response.content:
        if block.type == "tool_use":
            print(f"  tool_use: {block.name}({json.dumps(block.input)})")

banner("Final answer")
print(text_of(response))

# Bonus: a server-side tool (Anthropic runs it; you just declare it).
banner("Bonus – server tool: web search")
r = create(
    max_tokens=4000,
    tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 2}],
    messages=[{"role": "user", "content": "What is the latest stable MLflow release? One line, cite the source."}],
)
print(text_of(r))
