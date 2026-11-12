"""Lesson 23 – measure the main cost/latency levers.

Run:  python level3_advanced_agent_skills/23_optimize_performance/optimize.py [caching|effort|routing|parallel|streaming]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import MODEL, banner, client, create, require_api_key, text_of
from common.corpus import load_documents

# $ per 1M tokens (input, output) – list prices at time of writing; verify on the pricing page.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.1


def cost(model: str, usage) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (usage.input_tokens * pin + cw * pin * CACHE_WRITE_MULT + cr * pin * CACHE_READ_MULT
            + usage.output_tokens * pout) / 1e6


def report(label: str, model: str, r, seconds: float):
    u = r.usage
    print(f"{label:28} {seconds:6.1f}s  in={u.input_tokens:>6} cache_w={u.cache_creation_input_tokens or 0:>6} "
          f"cache_r={u.cache_read_input_tokens or 0:>6} out={u.output_tokens:>5}  ${cost(model, u):.4f}")


def timed(**kwargs):
    t0 = time.time()
    r = create(**kwargs)
    return r, time.time() - t0


# ------------------------------------------------------------------ 1. prompt caching
def caching():
    banner("1) Prompt caching – the whole repo as a cached system prompt")
    repo = "\n\n".join(f'<file path="{p}">\n{t}\n</file>' for p, t in load_documents().items())
    system = [
        {"type": "text", "text": "You are an expert on this MLOps repository. Answer briefly."},
        {"type": "text", "text": repo, "cache_control": {"type": "ephemeral"}},  # stable prefix -> cached
    ]
    for i, q in enumerate(["Which model is trained?", "Where is it deployed?", "Which secrets does CI need?"]):
        r, s = timed(system=system, max_tokens=1000, output_config={"effort": "low"},
                     messages=[{"role": "user", "content": q}])
        report(f"call {i + 1} ({'write' if i == 0 else 'read'} cache)", MODEL, r, s)
    print("→ calls 2-3 read the big prefix from cache: much cheaper input and usually faster.")


# ------------------------------------------------------------------ 2. effort
def effort():
    banner("2) Effort – same task, different thinking depth")
    task = "Give 3 risks of deploying an ML model without monitoring. One line each."
    for level in ("low", "medium", "high"):
        r, s = timed(max_tokens=8000, output_config={"effort": level}, messages=[{"role": "user", "content": task}])
        report(f"effort={level}", MODEL, r, s)


# ------------------------------------------------------------------ 3. model routing
def routing():
    banner("3) Model routing – simple classification on different model tiers")
    msg = "Classify the sentiment of this review as positive/negative/neutral. Reply with one word: " \
          "'The resort spa was lovely but the transfer was late.'"
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", MODEL):
        r, s = timed(model=model, max_tokens=1000, output_config={"effort": "low"},
                     messages=[{"role": "user", "content": msg}])
        report(f"{model} -> {text_of(r)[:10]}", model, r, s)
    print("→ if the cheap model's answers pass your eval, route this step to it.")


# ------------------------------------------------------------------ 4. parallelism
def parallel():
    banner("4) Sequential vs parallel independent calls")
    qs = [f"In one sentence, what does the MLOps term '{t}' mean?" for t in ("drift", "feature store", "canary", "model registry")]
    call = lambda q: create(max_tokens=500, output_config={"effort": "low"}, messages=[{"role": "user", "content": q}])
    t0 = time.time()
    for q in qs:
        call(q)
    seq = time.time() - t0
    t0 = time.time()
    with ThreadPoolExecutor(len(qs)) as pool:
        list(pool.map(call, qs))
    par = time.time() - t0
    print(f"sequential: {seq:.1f}s   parallel: {par:.1f}s   speed-up: {seq / par:.1f}x")


# ------------------------------------------------------------------ 5. streaming
def streaming():
    banner("5) Streaming – time to first token vs total time")
    t0, first = time.time(), None
    with client.messages.stream(model=MODEL, max_tokens=2000, output_config={"effort": "low"},
                                messages=[{"role": "user", "content": "Write 8 MLOps best practices, one line each."}]) as stream:
        for _ in stream.text_stream:
            first = first or time.time() - t0
    print(f"first token after {first:.1f}s, complete after {time.time() - t0:.1f}s → users see progress immediately")


EXPERIMENTS = {"caching": caching, "effort": effort, "routing": routing, "parallel": parallel, "streaming": streaming}

if __name__ == "__main__":
    require_api_key()
    for name in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[name]()
