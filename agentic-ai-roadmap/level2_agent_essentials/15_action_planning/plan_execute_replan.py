"""Lesson 15 – plan-and-execute with retries and re-planning.

Run:  python level2_agent_essentials/15_action_planning/plan_execute_replan.py ["goal"]
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ask, banner, create, parse, require_api_key, text_of
from common.tools import calculator, conversion_rate_by, dataset_columns, dataset_stats, execute, schema

TOOLS = [schema(f) for f in (dataset_columns, dataset_stats, conversion_rate_by, calculator)]
MAX_REPLANS = 2


class Step(BaseModel):
    id: int
    action: str
    success_criterion: str


class Plan(BaseModel):
    steps: list[Step]


class StepResult(BaseModel):
    success: bool
    findings: str


def make_plan(goal: str, learned: str = "") -> Plan:
    return parse(
        Plan,
        f"<goal>{goal}</goal>\n<learned_so_far>{learned or 'nothing yet'}</learned_so_far>\n"
        "Create a short plan (3-5 steps) of data-analysis steps. The data is a wellness-tourism customer table "
        "with target ProdTaken; tools can list columns, describe a column, compute conversion by a column, and "
        "do arithmetic. The last step should produce the deliverable. Skip steps already covered by what was learned.",
        max_tokens=3000,
    )


def run_step(step: Step, context: str, max_tool_rounds: int = 6) -> StepResult:
    """A small executor agent that works on ONE step."""
    messages = [{"role": "user", "content":
                 f"<progress>{context}</progress>\n<step>{step.action}</step>\n"
                 f"<success_criterion>{step.success_criterion}</success_criterion>\n"
                 "Do this step with the tools. If a tool fails, adapt (e.g. check column names)."}]
    for _ in range(max_tool_rounds):
        r = create(tools=TOOLS, messages=messages, max_tokens=8000)
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            break
        results = []
        for b in r.content:
            if b.type == "tool_use":
                out, err = execute(b.name, b.input)
                print(f"     {'❌' if err else '🔧'} {b.name}({json.dumps(b.input)}) -> {out[:120]}")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    # Self-assessment against the success criterion, as structured output.
    # The history contains tool_use blocks, so tools must still be declared; tool_choice "none" stops new calls.
    return parse(StepResult, messages=messages + [{"role": "user", "content":
                 "Did you meet the success criterion? Summarise the concrete findings (numbers) in 1-3 sentences."}],
                 tools=TOOLS, tool_choice={"type": "none"}, max_tokens=2000)


def plan_and_execute(goal: str) -> str:
    plan, progress, replans = make_plan(goal), [], 0
    queue = list(plan.steps)
    print("PLAN:\n" + "\n".join(f"  [{s.id}] {s.action}  ✓ when: {s.success_criterion}" for s in queue))

    while queue:
        step = queue.pop(0)
        print(f"\n▶ step {step.id}: {step.action}")
        result = run_step(step, "\n".join(progress))
        if not result.success:  # retry once with the failure as context
            print(f"   ↻ retrying – {result.findings}")
            result = run_step(step, "\n".join(progress) + f"\nPREVIOUS ATTEMPT FAILED: {result.findings}")
        progress.append(f"step {step.id} ({'ok' if result.success else 'FAILED'}): {result.findings}")
        print(f"   {'✅' if result.success else '⚠️'} {result.findings}")

        if not result.success and replans < MAX_REPLANS:  # re-plan the rest with what we learned
            replans += 1
            queue = make_plan(goal, "\n".join(progress)).steps
            print(f"   🔁 re-planned ({replans}/{MAX_REPLANS}): " + " | ".join(s.action for s in queue))

    return ask(f"<goal>{goal}</goal>\n<findings>\n" + "\n".join(progress) + "\n</findings>\n"
               "Write the deliverable using only these findings.", max_tokens=3000)


if __name__ == "__main__":
    require_api_key()
    goal = " ".join(sys.argv[1:]) or (
        "Write a 5-bullet sales-targeting brief: which segments (age, occupation, city tier, and the "
        "'LoyaltyTier' column) convert best, with numbers."  # LoyaltyTier doesn't exist -> forces adaptation
    )
    banner(f"GOAL: {goal}")
    banner("DELIVERABLE")
    print(plan_and_execute(goal))
