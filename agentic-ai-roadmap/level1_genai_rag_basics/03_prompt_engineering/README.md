# 03 · Prompt Engineering

> **Goal:** Write prompts that get reliable, well-formatted results.

## The patterns that matter most

| Pattern | Why | Example |
|---|---|---|
| **Be clear & specific** | The model can't read your mind. State the task, audience, format, length. | "Summarise for a CFO in 3 bullets, ≤ 15 words each." |
| **Give context / the *why*** | Explaining the goal beats piling on rules. | "This goes in a Slack alert, so keep it scannable." |
| **System prompt = role** | Sets persona, constraints, tone for the whole conversation. | `system="You are an MLOps reviewer…"` |
| **XML tags** | Separate instructions from data; makes injection harder and parsing easier. | `<code>…</code>`, `<question>…</question>` |
| **Few-shot examples** | Show 2-3 input→output pairs; the model copies the pattern. | See `few_shot()` in the script |
| **Let it think** | For reasoning tasks, ask for step-by-step reasoning (or raise `effort`). | "Think it through, then give the answer in `<answer>` tags." |
| **Structured output** | Ask for JSON with a schema — or enforce it with `output_config.format` (lesson 18 uses Pydantic). | See `structured()` |
| **Prompt chaining** | Split a big task into smaller prompts (→ lesson 13, workflows). | |

## Anti-patterns
- Vague asks ("make it better") · Contradictory rules · SHOUTING "MUST/NEVER" everywhere (modern models over-react to it) · Stuffing untrusted user text into the system prompt.

## Hands-on
```bash
python level1_genai_rag_basics/03_prompt_engineering/prompt_patterns.py
```
Compares a vague vs. a specific prompt, then demos system role, XML tags, few-shot, step-by-step, and schema-enforced JSON — all on code from this repo.

## Exercises
1. Write a prompt that turns `deployment/Dockerfile` into a numbered explanation for a junior dev.
2. Build a few-shot classifier that labels commit messages as `feat` / `fix` / `docs` / `refactor`.
3. Make the JSON extractor in the script also return `risk_level: "low"|"medium"|"high"`.
