# 09 · Tool Integration (function calling)

> **Goal:** Let the LLM *call your code* — the single most important building block of agents.

## How tool use works

```
1. You send: messages + tool definitions (name, description, JSON schema)
2. Model replies: stop_reason="tool_use" + a tool_use block {name, input, id}
3. YOUR code runs the function with that input
4. You send back: a tool_result block {tool_use_id, content}
5. Model replies with the final answer (or asks for another tool)
```

The model never executes anything itself — it *asks*, your code *decides and runs*. That's also your security boundary.

## Writing good tools
- **Name & description are the prompt.** Say what it does, when to use it, what it returns, and its limits.
- **Tight schemas**: `enum`s, `required`, `additionalProperties: false`; add `strict: true` to guarantee valid inputs.
- **Return useful errors** with `is_error: true` ("City 'xyz' not found – try a full city name") so the model can recover.
- **Return compact results** – tool output costs tokens on every subsequent turn.
- **Parallel calls**: the model may request several tools at once; run them and return *all* results in one message.

## Built-in (server) tools
Anthropic also hosts tools you just declare – **web search**, **web fetch**, **code execution** – no implementation needed (demoed at the end of the script).

## Hands-on
```bash
python level1_genai_rag_basics/09_tool_integration/first_tool_call.py
```
One request/response cycle done **by hand** with two tools over this repo's dataset: `get_dataset_stats` and `calculator`. Read every printed step — lesson 11 turns this into a loop.

## Exercises
1. Add a `search_repo(query)` tool backed by the lesson-06 retriever.
2. Make `get_dataset_stats` return an error for unknown columns and watch the model recover.
3. Ask a question that needs both tools in parallel. How many `tool_use` blocks come back?
