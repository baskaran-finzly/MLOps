# 02 · Basics of LLMs

> **Goal:** Understand how LLMs are trained and the knobs you control when you use one.

## How an LLM is built
1. **Pre-training** – predict the next token on trillions of tokens of text/code. The model learns language, facts, and reasoning patterns.
2. **Post-training** – instruction tuning + RLHF / Constitutional AI turn the raw predictor into a helpful, honest, harmless assistant.
3. **Inference** – you send tokens in, it generates tokens out, one at a time.

## Core vocabulary

| Concept | What you need to know |
|---|---|
| **Token** | ~¾ of an English word. Billing, limits and speed are all measured in tokens. |
| **Context window** | Max tokens (input + output) the model can see at once. Current Claude models: 1M. |
| **`max_tokens`** | Hard cap on *output* length. If hit, `stop_reason == "max_tokens"` and the answer is cut off. |
| **Stateless API** | The model remembers nothing between calls. *You* resend the conversation history every time (→ lesson 14, memory). |
| **Thinking / effort** | Modern models reason before answering. You control depth with `output_config={"effort": "low" \| "medium" \| "high" \| "xhigh" \| "max"}` instead of temperature. |
| **Knowledge cutoff** | The model only knows what was in its training data → RAG (lesson 06) and tools (lesson 09) bring in fresh/private data. |
| **Hallucination** | Fluent but false output. Mitigate with grounding (RAG), citations, and evaluation. |

> **Note on temperature:** older tutorials tune `temperature` / `top_p`. Current Claude models don't accept sampling parameters — use `effort` and clear prompts instead.

## Hands-on
```bash
python level1_genai_rag_basics/02_llm_basics/tokens_and_context.py
```
Shows: counting tokens before sending, multi-turn history (statelessness), `max_tokens` truncation, and effort levels.

## Exercises
1. Count tokens for `model_building/train.py`. How many could fit in a 1M context?
2. Remove the history from the multi-turn demo — does Claude still know your name?
3. Compare answer length/quality at `effort="low"` vs `"high"` on a tricky logic puzzle.
