"""Lesson 22 – an autonomous agent that works through a persistent task queue.

Run:  python level3_advanced_agent_skills/22_autonomous_loops/autonomous_task_agent.py [--reset] [--watch SECONDS]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anthropic import beta_tool

from common import MODEL, banner, client, fallback_kwargs, require_api_key, text_of
from common import tools as T

HERE = Path(__file__).parent
TASKS = HERE / "tasks.json"
REPORTS = HERE / "reports"
STOP_FILE = HERE / "STOP"
MAX_TASKS_PER_RUN = 3
MAX_ATTEMPTS = 2
MAX_NEW_TASKS_PER_RUN = 3

SEED_TASKS = [
    "Profile data quality: report missing values and suspicious outliers for every numeric column.",
    "Find the 3 customer segments with the highest conversion rate (use at least 3 different columns).",
    "Review model_building/train.py and list how the model is evaluated and which metric selects the best model.",
    "Check .github/workflows/pipeline.yml and list what would break if the HF_TOKEN secret were missing.",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    return json.loads(TASKS.read_text())


def save(state: dict) -> None:  # checkpoint after every change
    TASKS.write_text(json.dumps(state, indent=2))


def reset() -> None:
    save({"tasks": [{"id": i + 1, "description": d, "status": "pending", "attempts": 0, "result": None,
                     "updated_at": now()} for i, d in enumerate(SEED_TASKS)], "runs": []})
    print(f"Seeded {len(SEED_TASKS)} tasks into {TASKS}")


# ------------------------------------------------------------------ tools the agent gets for queue management
_new_tasks_this_run = 0


@beta_tool
def add_follow_up_task(description: str) -> str:
    """Add a follow-up task to the queue for a future run. Only for genuinely important next steps.

    Args:
        description: A clear, self-contained task description.
    """
    global _new_tasks_this_run
    if _new_tasks_this_run >= MAX_NEW_TASKS_PER_RUN:
        return "Not added: follow-up task limit for this run reached."
    state = load()
    if any(t["description"].lower() == description.lower() for t in state["tasks"]):
        return "Not added: an identical task already exists."
    new_id = max(t["id"] for t in state["tasks"]) + 1
    state["tasks"].append({"id": new_id, "description": description, "status": "pending", "attempts": 0,
                           "result": None, "updated_at": now(), "created_by": "agent"})
    save(state)
    _new_tasks_this_run += 1
    return f"Added task #{new_id}."


SYSTEM = """You are an autonomous MLOps analyst working through a task queue without a human present.
Complete the given task using the tools, then reply with a concise markdown report (findings with numbers / file
names). If you notice an important next step, you may add ONE follow-up task. If the task is impossible, start your
reply with 'BLOCKED:' and explain why."""


def work_on(task: dict) -> tuple[str, str]:
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000, system=SYSTEM, max_iterations=15,
        tools=[*T.runner_tools(), add_follow_up_task],
        messages=[{"role": "user", "content": f"Task #{task['id']}: {task['description']}"}],
        **fallback_kwargs(),
    )
    final = None
    for message in runner:
        for b in message.content:
            if b.type == "tool_use":
                print(f"   🔧 {b.name}({json.dumps(b.input)[:100]})")
        final = message
    report = text_of(final)
    status = "blocked" if report.startswith("BLOCKED:") or final.stop_reason in ("max_tokens", "refusal") else "done"
    return status, report


def run_once() -> None:
    global _new_tasks_this_run
    _new_tasks_this_run = 0
    REPORTS.mkdir(exist_ok=True)
    state = load()
    for t in state["tasks"]:  # crash recovery: a task left in_progress by a dead run goes back to the queue
        if t["status"] == "in_progress":
            t["status"] = "pending"
    save(state)

    processed, tried = 0, set()
    while processed < MAX_TASKS_PER_RUN:
        if STOP_FILE.exists():
            print("🛑 STOP file found – halting.")
            break
        state = load()  # always re-read: the agent may have added tasks
        task = next((t for t in state["tasks"] if t["status"] == "pending" and t["id"] not in tried), None)
        if task is None:
            break
        tried.add(task["id"])
        print(f"\n▶ Task #{task['id']}: {task['description']}")
        task["status"], task["attempts"], task["updated_at"] = "in_progress", task["attempts"] + 1, now()
        save(state)
        try:
            status, report = work_on(task)
        except Exception as exc:  # network/API failure: retry next run, block after MAX_ATTEMPTS
            status, report = ("blocked" if task["attempts"] >= MAX_ATTEMPTS else "pending"), f"ERROR: {exc}"
        state = load()  # reload: the agent may have appended tasks
        task = next(t for t in state["tasks"] if t["id"] == task["id"])
        task.update(status=status, updated_at=now(), result=f"reports/task_{task['id']}.md")
        (REPORTS / f"task_{task['id']}.md").write_text(f"# Task {task['id']}\n\n{task['description']}\n\n{report}\n")
        save(state)
        processed += 1
        print(f"   → {status}")

    state = load()
    state["runs"].append({"at": now(), "processed": processed, "new_tasks": _new_tasks_this_run})
    save(state)
    counts = {s: sum(t["status"] == s for t in state["tasks"]) for s in ("pending", "done", "blocked")}
    banner(f"Run finished: processed={processed}  queue={counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="seed tasks.json and exit")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="keep running every N seconds")
    args = ap.parse_args()
    if args.reset or not TASKS.exists():
        reset()
        if args.reset:
            sys.exit()
    require_api_key()
    while True:
        run_once()
        if not args.watch or STOP_FILE.exists() or not any(t["status"] == "pending" for t in load()["tasks"]):
            break
        print(f"Sleeping {args.watch}s… (touch {STOP_FILE.name} to stop)")
        time.sleep(args.watch)
