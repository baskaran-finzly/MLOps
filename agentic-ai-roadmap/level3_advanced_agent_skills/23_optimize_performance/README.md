# 23 · Optimize Performance (speed, cost, error handling)

> **Goal:** Make agents fast, cheap and robust — measured, not guessed.

## Where cost & latency come from
`cost ≈ Σ over every turn (input tokens × in-price + output tokens × out-price)` — and an agent **resends the whole history every turn**, so input tokens grow fast. Latency ≈ number of sequential turns × (time to first token + output tokens ÷ speed).

## Levers, in the order to try them

| # | Lever | Effect | How |
|---|---|---|---|
| 1 | **Prompt caching** | Up to ~90 % cheaper & faster on repeated prefixes (system prompt, tools, docs, history) | `cache_control` on the stable prefix; keep it byte-identical (no timestamps!) |
| 2 | **Token hygiene** | Fewer input tokens every turn | Truncate/summarise tool outputs, return only needed fields, context editing/compaction for long runs |
| 3 | **Effort tuning** | Fewer thinking/output tokens on easy steps | `output_config={"effort": "low"}` for routing/classification; higher for hard reasoning |
| 4 | **Fewer round trips** | Lower latency | Parallel tool calls, batch several questions per call, workflows instead of open loops where possible |
| 5 | **Parallelism** | Lower wall-clock | Run independent LLM calls/sub-agents concurrently |
| 6 | **Model routing** | Cheaper per token | Small model (Haiku) for simple high-volume steps, strongest model where quality matters — *measure quality first* |
| 7 | **Batch API** | 50 % cheaper | Non-urgent bulk jobs (nightly scoring, evals) via `client.messages.batches` |
| 8 | **Streaming** | Better *perceived* latency | Show tokens as they arrive |

## Error handling for production
- SDK auto-retries 408/409/429/5xx with backoff (`max_retries`); set **timeouts**.
- Catch specific errors (`RateLimitError`, `APIConnectionError`, `BadRequestError`) — retry only transient ones.
- Handle every `stop_reason`: `end_turn`, `tool_use`, `max_tokens` (raise limit / continue), `pause_turn` (resend), `refusal` (fallback/message).
- Tool failures → `is_error: true` results, not crashes. Add circuit breakers for flaky external APIs.
- Always cap steps, tokens and wall-clock time.

## Hands-on
```bash
python level3_advanced_agent_skills/23_optimize_performance/optimize.py            # all experiments
python level3_advanced_agent_skills/23_optimize_performance/optimize.py caching    # one experiment
```
Experiments: `caching`, `effort`, `routing`, `parallel`, `streaming` — each prints latency, tokens and estimated cost so you can make data-driven choices.

> Prices in the script are list prices at the time of writing — check the current pricing page before relying on them.

## Exercises
1. Add caching to the lesson-11 agent (system + tools) and measure cost over a 10-step run.
2. Use the Batch API to run the lesson-18 eval cases at half price.
3. Find the cheapest model + effort combination that still passes your lesson-18 eval at ≥ 90 %.
