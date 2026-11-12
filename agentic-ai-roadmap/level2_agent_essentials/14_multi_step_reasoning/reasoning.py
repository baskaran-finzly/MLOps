"""Lesson 14 – multi-step reasoning techniques.

Run:  python level2_agent_essentials/14_multi_step_reasoning/reasoning.py
"""

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ask, banner, create, parse, require_api_key, text_of
from common.tools import calculator, conversion_rate_by

require_api_key()

PUZZLE = ("A sales team pitches 3 packages. Basic converts 30% of 1,000 leads. Deluxe converts 12% of 800. "
          "Standard converts 18% of 600. If the company moves 200 Deluxe leads to Basic (and Basic keeps its rate), "
          "how many more total sales do they get? Final answer as a number on the last line.")

# 1) Adaptive thinking + effort, with a visible thinking summary.
banner("1) Thinking depth via effort")
for effort in ("low", "high"):
    r = create(
        max_tokens=8000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": PUZZLE}],
    )
    thoughts = " ".join(b.thinking for b in r.content if b.type == "thinking").strip()
    print(f"\n--- effort={effort}  output_tokens={r.usage.output_tokens}")
    if thoughts:
        print(f"thinking summary: {thoughts[:500]}…")
    print(f"answer: {text_of(r).splitlines()[-1] if text_of(r) else ''}")


# 2) Decomposition: break a big question into sub-questions, answer each with data, combine.
class Plan(BaseModel):
    sub_questions: list[str]


banner("2) Decomposition")
big_q = "Should marketing focus on younger customers or on customers with passports?"
plan = parse(Plan, f"Break this into 2-4 data sub-questions answerable from a customer dataset with columns "
                   f"Age, Passport, ProdTaken:\n{big_q}", max_tokens=1500, output_config={"effort": "low"})
evidence = {"by_age": conversion_rate_by("Age"), "by_passport": conversion_rate_by("Passport")}
for i, sq in enumerate(plan.sub_questions, 1):
    print(f"  sub-question {i}: {sq}")
print("\n" + ask(f"<evidence>{evidence}</evidence>\n<sub_questions>{plan.sub_questions}</sub_questions>\n"
                 f"Answer each sub-question from the evidence, then answer: {big_q}", max_tokens=3000))

# 3) Self-consistency: several independent samples, majority vote.
banner("3) Self-consistency (3 samples, majority vote)")


def sample(_):
    out = ask(PUZZLE + " Reply with ONLY the number.", max_tokens=4000, output_config={"effort": "low"})
    return out.strip().replace(",", "")


with ThreadPoolExecutor(3) as pool:
    votes = list(pool.map(sample, range(3)))
print(f"votes={votes} -> majority={Counter(votes).most_common(1)[0][0]}")

# 4) Self-verification with a tool: recompute the arithmetic exactly.
banner("4) Verify with a tool")
truth = calculator("((1000 + 200) * 0.30 + (800 - 200) * 0.12 + 600 * 0.18) - (1000 * 0.30 + 800 * 0.12 + 600 * 0.18)")
print(f"calculator says the gain is {truth}; compare with the votes above.")
