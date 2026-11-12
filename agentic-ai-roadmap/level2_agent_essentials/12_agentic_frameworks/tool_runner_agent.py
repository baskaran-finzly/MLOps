"""Lesson 12 – the lesson-11 agent rebuilt with the Anthropic SDK tool runner.

The runner owns the loop: it calls the API, runs your decorated functions,
sends results back, and stops when the model is done. You still get to see
(and intercept) every turn.

Run:  python level2_agent_essentials/12_agentic_frameworks/tool_runner_agent.py ["goal"]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import MODEL, banner, client, fallback_kwargs, require_api_key, text_of
from common.tools import calculator, conversion_rate_by, dataset_columns, dataset_stats, runner_tools

SYSTEM = """You are a data analyst for a travel company selling wellness-tourism packages.
Use the tools to inspect the customer dataset (target: ProdTaken = 1 if bought).
Use the calculator for arithmetic. Answer concisely with the key numbers."""

if __name__ == "__main__":
    require_api_key()
    goal = " ".join(sys.argv[1:]) or "Do customers with a passport buy more often? Quantify the difference in percentage points."
    banner(f"GOAL: {goal}")

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=runner_tools(dataset_columns, dataset_stats, conversion_rate_by, calculator),
        messages=[{"role": "user", "content": goal}],
        max_iterations=12,  # safety cap, like MAX_STEPS in lesson 11
        **fallback_kwargs(),
    )

    # Each iteration yields the assistant message *before* its tools run – a hook for logging/approval.
    final = None
    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                print(f"🔧 {block.name}({block.input})")
        final = message

    banner("ANSWER")
    print(text_of(final))
