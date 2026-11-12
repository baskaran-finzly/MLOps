"""Lesson 11 – a ReAct agent with a hand-written loop.

Run:  python level2_agent_essentials/11_build_first_agent/data_analyst_agent.py ["goal"]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import banner, create, require_api_key, text_of
from common.tools import calculator, conversion_rate_by, dataset_columns, dataset_stats, execute, schema

SYSTEM = """You are a data analyst for a travel company selling wellness-tourism packages.
You have tools to inspect the customer dataset (target column: ProdTaken = 1 if the customer bought).
Work step by step: discover the data, run the analyses you need, use the calculator for any arithmetic,
then give a concise, actionable answer with the key numbers. Don't invent numbers you didn't get from a tool."""

TOOLS = [schema(f) for f in (dataset_columns, dataset_stats, conversion_rate_by, calculator)]
MAX_STEPS = 12


def run_agent(goal: str, max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    messages = [{"role": "user", "content": goal}]
    usage = {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0}

    for step in range(1, max_steps + 1):
        response = create(system=SYSTEM, tools=TOOLS, messages=messages, output_config={"effort": "medium"})
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})  # always the full content

        if response.stop_reason == "pause_turn":  # server paused a long turn; just resend
            continue
        if response.stop_reason != "tool_use":  # end_turn / max_tokens / refusal -> stop
            return {"answer": text_of(response), "steps": step, "usage": usage, "messages": messages}

        results = []
        for block in response.content:
            if block.type == "text" and block.text.strip() and verbose:
                print(f"\n💭 {block.text.strip()}")
            if block.type == "tool_use":
                output, is_error = execute(block.name, block.input)
                usage["tool_calls"] += 1
                if verbose:
                    flag = "❌" if is_error else "🔧"
                    print(f"{flag} step {step}: {block.name}({json.dumps(block.input)}) -> {output[:160]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    # Out of steps: ask for a best-effort answer without tools.
    messages.append({"role": "user", "content": "You've hit the step limit. Give your best answer now from what you have."})
    final = create(system=SYSTEM, tools=TOOLS, tool_choice={"type": "none"}, messages=messages)
    return {"answer": text_of(final), "steps": max_steps, "usage": usage, "messages": messages}


if __name__ == "__main__":
    require_api_key()
    goal = " ".join(sys.argv[1:]) or (
        "Which age group and which occupation have the highest purchase rate? "
        "By what percentage is the best age group's rate higher than the overall rate?"
    )
    banner(f"GOAL: {goal}")
    result = run_agent(goal)
    banner("ANSWER")
    print(result["answer"])
    u = result["usage"]
    print(f"\nsteps={result['steps']} tool_calls={u['tool_calls']} tokens in/out={u['input_tokens']}/{u['output_tokens']}")
