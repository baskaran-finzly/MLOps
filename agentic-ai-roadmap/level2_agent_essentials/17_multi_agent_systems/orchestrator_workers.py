"""Lesson 17 – orchestrator–workers multi-agent system.

The lead agent has ONE tool, `delegate`, which spins up a specialist sub-agent
(its own system prompt, own tools, own context) and returns only its findings.
Several delegate calls in one turn run in parallel.

Run:  python level2_agent_essentials/17_multi_agent_systems/orchestrator_workers.py ["task"]
      WORKER_MODEL=claude-haiku-5-5 python ...   # cheaper workers
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anthropic import beta_tool

from common import MODEL, banner, client, create, fallback_kwargs, require_api_key, text_of
from common import tools as T

WORKER_MODEL = os.getenv("WORKER_MODEL", MODEL)

SPECIALISTS = {
    "data_analyst": {
        "system": "You are a data analyst. Investigate the wellness-tourism customer dataset with your tools: "
                  "data quality (missing values, outliers), class balance of ProdTaken, strongest segments. "
                  "Return at most 8 bullet findings with numbers.",
        "tools": [T.dataset_columns, T.dataset_stats, T.conversion_rate_by, T.calculator],
    },
    "code_auditor": {
        "system": "You are a senior ML engineer auditing training code. Use the repo tools to read "
                  "model_building/*.py. Check for data leakage, class-imbalance handling, evaluation metrics, "
                  "reproducibility. Return at most 8 bullet findings, citing file names.",
        "tools": [T.list_repo_files, T.read_repo_file, T.search_repo],
    },
    "mlops_reviewer": {
        "system": "You are an MLOps reviewer. Use the repo tools to inspect the CI pipeline "
                  "(.github/workflows/pipeline.yml), deployment/ and hosting/. Check automation, secrets handling, "
                  "monitoring, rollback. Return at most 8 bullet findings, citing file names.",
        "tools": [T.list_repo_files, T.read_repo_file, T.search_repo],
    },
}


def run_worker(role: str, brief: str) -> str:
    spec = SPECIALISTS[role]
    runner = client.beta.messages.tool_runner(
        model=WORKER_MODEL, max_tokens=16000, system=spec["system"], max_iterations=10,
        tools=T.runner_tools(*spec["tools"]), messages=[{"role": "user", "content": brief}],
        **fallback_kwargs(WORKER_MODEL),
    )
    final = runner.until_done()
    print(f"   ✅ {role} finished")
    return text_of(final)


@beta_tool
def delegate(role: Literal["data_analyst", "code_auditor", "mlops_reviewer"], brief: str) -> str:
    """Delegate a sub-task to a specialist agent and get back its findings.

    Call several delegates in the same turn to run them in parallel.

    Args:
        role: Which specialist: data_analyst (dataset), code_auditor (training code), mlops_reviewer (CI/deploy).
        brief: Complete instructions: goal, what to check, and what to return. The specialist knows nothing else.
    """
    return run_worker(role, brief)


LEAD_SYSTEM = """You are the lead of an ML readiness review. Break the task into sub-tasks and delegate them
to specialists with complete briefs (they share none of your context). Delegate independent sub-tasks in the
same turn so they run in parallel. Then write the final report: verdict (Ready / Ready with fixes / Not ready),
top risks, and a prioritised action list. Base it only on specialists' findings."""


def run_lead(task: str) -> str:
    """Hand-written lead loop so we can run parallel delegate calls concurrently."""
    tools = [delegate.to_dict()]
    messages = [{"role": "user", "content": task}]
    for _ in range(6):
        r = create(system=LEAD_SYSTEM, tools=tools, messages=messages, output_config={"effort": "high"})
        messages.append({"role": "assistant", "content": r.content})
        calls = [b for b in r.content if b.type == "tool_use"]
        if r.stop_reason != "tool_use" or not calls:
            return text_of(r)
        for c in calls:
            print(f"📨 lead → {c.input['role']}: {c.input['brief'][:110]}…")
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:  # workers in parallel
            outputs = list(pool.map(lambda c: run_worker(c.input["role"], c.input["brief"]), calls))
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out} for c, out in zip(calls, outputs)
        ]})
    return "Lead agent hit its turn limit."


if __name__ == "__main__":
    require_api_key()
    task = " ".join(sys.argv[1:]) or (
        "Assess whether our wellness-tourism purchase-prediction model, its data and its pipeline are ready "
        "to drive a targeted marketing campaign next month."
    )
    banner(f"TASK: {task}\n(lead={MODEL}, workers={WORKER_MODEL})")
    report = run_lead(task)
    banner("FINAL REPORT")
    print(report)
