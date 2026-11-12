"""Lesson 18 – an evaluation harness for the lesson-11 Data Analyst Agent.

Run:  python level2_agent_essentials/18_agent_evaluation/eval_harness.py [--repeats N] [--mlflow]
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "level2_agent_essentials" / "11_build_first_agent"))

from common import banner, parse, require_api_key
from common.tools import _df
from data_analyst_agent import run_agent  # the agent under test

# ---------------------------------------------------------------- ground truth from the source of truth
df = _df()
overall_rate = df["ProdTaken"].mean()
passport_count = int((df["Passport"] == 1).sum())
occ = df.groupby("Occupation")["ProdTaken"].agg(["mean", "count"])
best_occupation = occ[occ["count"] >= 100]["mean"].idxmax()
avg_age_buyers = df.loc[df["ProdTaken"] == 1, "Age"].mean()


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def has_number(expected: float, rel_tol: float = 0.02) -> Callable[[str], bool]:
    """Pass if any number in the answer is within tolerance (also accepts % form of a fraction)."""
    def check(answer: str) -> bool:
        targets = [expected] + ([expected * 100] if 0 < expected < 1 else [])
        return any(abs(n - t) <= rel_tol * max(abs(t), 1e-9) for n in numbers_in(answer) for t in targets)
    return check


def contains(*words: str) -> Callable[[str], bool]:
    return lambda answer: all(w.lower() in answer.lower() for w in words)


@dataclass
class Case:
    id: str
    question: str
    check: Callable[[str], bool] | None  # code-based grader (None = judge only)
    reference: str  # given to the LLM judge
    required_tools: set = field(default_factory=set)


CASES = [
    Case("overall_rate", "What fraction of customers bought a package?",
         has_number(overall_rate), f"{overall_rate:.3f} ({overall_rate:.1%})", {"dataset_stats", "conversion_rate_by"}),
    Case("passport_count", "How many customers have a passport?",
         has_number(passport_count, 0.005), str(passport_count)),
    Case("best_occupation", "Among occupations with at least 100 customers, which has the highest purchase rate?",
         contains(best_occupation), best_occupation, {"conversion_rate_by"}),
    Case("avg_age_buyers", "What is the average age of customers who bought a package?",
         has_number(avg_age_buyers, 0.03), f"{avg_age_buyers:.1f}"),
    Case("unanswerable", "What is the average credit score of our customers?",
         None, "The dataset has no credit score column; a good answer says so and does not invent a number."),
]


# ---------------------------------------------------------------- LLM-as-judge
class Verdict(BaseModel):
    reasoning: str
    correct: bool
    faithful: bool  # no invented facts/numbers
    score: int  # 1-5 overall quality


def judge(case: Case, answer: str) -> Verdict:
    return parse(
        Verdict,
        f"You are grading an analytics agent.\n<question>{case.question}</question>\n"
        f"<reference_answer>{case.reference}</reference_answer>\n<agent_answer>{answer}</agent_answer>\n"
        "Is the agent's answer correct per the reference? Is it faithful (no invented numbers/facts)? "
        "Give a 1-5 quality score. Reason briefly first.",
        max_tokens=3000,
        output_config={"effort": "low"},
    )


# ---------------------------------------------------------------- runner
def evaluate(case: Case) -> dict:
    t0 = time.time()
    result = run_agent(case.question, verbose=False)
    latency = time.time() - t0
    tools_used = {b.name for m in result["messages"] if m["role"] == "assistant"
                  for b in m["content"] if getattr(b, "type", None) == "tool_use"}
    code_pass = case.check(result["answer"]) if case.check else None
    trajectory_pass = not case.required_tools or bool(case.required_tools & tools_used)
    verdict = judge(case, result["answer"])
    passed = (code_pass is not False) and verdict.correct and verdict.faithful and trajectory_pass
    return {
        "case": case.id, "passed": passed, "code_pass": code_pass, "judge_correct": verdict.correct,
        "judge_faithful": verdict.faithful, "judge_score": verdict.score, "trajectory_pass": trajectory_pass,
        "tools_used": sorted(tools_used), "steps": result["steps"], "tool_calls": result["usage"]["tool_calls"],
        "input_tokens": result["usage"]["input_tokens"], "output_tokens": result["usage"]["output_tokens"],
        "latency_s": round(latency, 1), "answer": result["answer"], "judge_reasoning": verdict.reasoning,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()
    require_api_key()

    rows = []
    for case in CASES:
        for i in range(args.repeats):
            r = evaluate(case)
            r["run"] = i
            rows.append(r)
            print(f"{'✅' if r['passed'] else '❌'} {case.id:16} run{i} code={r['code_pass']} judge={r['judge_correct']}/"
                  f"{r['judge_faithful']} score={r['judge_score']} tools={r['tool_calls']} {r['latency_s']}s")

    n = len(rows)
    summary = {
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "pass_all_repeats": sum(all(r["passed"] for r in rows if r["case"] == c.id) for c in CASES) / len(CASES),
        "avg_judge_score": sum(r["judge_score"] for r in rows) / n,
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
    }
    banner("SCORECARD")
    for k, v in summary.items():
        print(f"  {k:18} {v:.3f}" if isinstance(v, float) else f"  {k:18} {v}")

    out = Path(__file__).with_name("eval_results.json")
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"\nDetailed results -> {out}")

    if args.mlflow:  # same experiment tracker the MLOps pipeline uses
        import mlflow

        mlflow.set_experiment("agent-evaluation")
        with mlflow.start_run(run_name="data-analyst-agent"):
            mlflow.log_metrics(summary)
            mlflow.log_artifact(str(out))
        print("Logged to MLflow experiment 'agent-evaluation'.")
