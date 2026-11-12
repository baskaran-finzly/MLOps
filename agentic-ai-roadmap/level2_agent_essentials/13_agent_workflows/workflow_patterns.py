"""Lesson 13 – workflow patterns: chaining, routing, parallelisation, evaluator-optimizer.

Run:  python level2_agent_essentials/13_agent_workflows/workflow_patterns.py [chaining|routing|parallel|evaluator]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import REPO_ROOT, ask, banner, parse, require_api_key
from common.retriever import TfidfRetriever, format_context
from common.tools import conversion_rate_by, dataset_columns

FAST = {"output_config": {"effort": "low"}}  # cheap/fast settings for simple steps


# ----------------------------------------------------------------------------- 1. chaining
def prompt_chaining():
    banner("1) Prompt chaining: commit log -> facts -> release note -> polish")
    commits = "29923d3 Updated changes with neat structure\n9a89867 Updated pipeline file with detailed explanation\n" \
              "4105be7 Refactored train script\n346f786 updated lib"
    facts = ask(f"List the concrete changes implied by these commits, one per line:\n{commits}", max_tokens=1000, **FAST)
    print("Step 1 – facts:\n" + facts)
    if not facts.strip():  # gate: stop early if a step produced nothing useful
        print("Gate failed – no facts extracted.")
        return
    draft = ask(f"Write a short release note (heading + bullets) from these changes:\n{facts}", max_tokens=1500)
    print("\nStep 2 – draft:\n" + draft)
    final = ask(f"Polish this release note for end users: plain language, max 80 words.\n\n{draft}", max_tokens=1500)
    print("\nStep 3 – final:\n" + final)


# ----------------------------------------------------------------------------- 2. routing
class Route(BaseModel):
    category: Literal["data", "code", "general"]
    reason: str


def route(question: str) -> str:
    decision = parse(
        Route,
        "Classify the question. data = about customer dataset numbers; code = about how this repo's "
        f"pipeline/code works; general = anything else.\n<question>{question}</question>",
        max_tokens=1000,
        **FAST,
    )
    print(f"  router -> {decision.category}  ({decision.reason})")
    if decision.category == "data":
        ctx = dataset_columns() + "\n" + conversion_rate_by("ProductPitched")
        return ask(f"<data>{ctx}</data>\nAnswer using the data: {question}", max_tokens=1500)
    if decision.category == "code":
        ctx = format_context(TfidfRetriever().search(question, k=4))
        return ask(f"<documents>{ctx}</documents>\nAnswer from the documents, citing files: {question}", max_tokens=2000)
    return ask(question, max_tokens=1000, **FAST)


def routing():
    banner("2) Routing")
    for q in ["Which product pitched has the best conversion?", "How is the model registered in MLflow?", "What is MLOps?"]:
        print(f"\nQ: {q}")
        print(route(q))


# ----------------------------------------------------------------------------- 3. parallelisation
def parallel():
    banner("3) Parallelisation (sectioning): 3 reviewers at once")
    code = (REPO_ROOT / "deployment" / "app.py").read_text()[:8000]
    aspects = {
        "security": "secrets handling, input validation, unsafe calls",
        "performance": "model loading, caching, repeated work",
        "readability": "naming, structure, comments",
    }

    def review(aspect: str, focus: str) -> str:
        return ask(
            f"Review this Streamlit app ONLY for {aspect} ({focus}). Give the top 2 findings, one line each.\n"
            f"<code>\n{code}\n</code>",
            max_tokens=1500,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {a: pool.submit(review, a, f) for a, f in aspects.items()}
        reviews = {a: f.result() for a, f in futures.items()}
    for a, r in reviews.items():
        print(f"\n[{a}]\n{r}")
    summary = ask("Merge these reviews into a prioritised top-3 action list:\n" +
                  "\n".join(f"<{a}>{r}</{a}>" for a, r in reviews.items()), max_tokens=1500)
    print("\n[aggregated]\n" + summary)


# ----------------------------------------------------------------------------- 5. evaluator-optimizer
class Critique(BaseModel):
    passed: bool
    feedback: list[str]


def evaluator_optimizer(max_rounds: int = 3):
    banner("5) Evaluator-optimizer: model card")
    criteria = ("Has sections: Intended use, Training data, Metrics, Limitations, Ethical considerations. "
                "Mentions XGBoost and the ProdTaken target. Under 200 words.")
    draft = ask("Write a short model card for an XGBoost model that predicts whether a customer "
                "buys a wellness-tourism package (target ProdTaken).", max_tokens=2000)
    for rnd in range(1, max_rounds + 1):
        critique = parse(Critique, f"<criteria>{criteria}</criteria>\n<draft>{draft}</draft>\n"
                                   "Does the draft meet ALL criteria? List concrete fixes if not.", max_tokens=1500)
        print(f"\nRound {rnd}: passed={critique.passed}  feedback={critique.feedback}")
        if critique.passed:
            break
        draft = ask(f"Revise the model card to address this feedback:\n{critique.feedback}\n\n<draft>{draft}</draft>\n"
                    "Return only the revised card.", max_tokens=2000)
    print("\nFinal model card:\n" + draft)


PATTERNS = {"chaining": prompt_chaining, "routing": routing, "parallel": parallel, "evaluator": evaluator_optimizer}

if __name__ == "__main__":
    require_api_key()
    chosen = sys.argv[1:] or list(PATTERNS)
    for name in chosen:
        PATTERNS[name]()
