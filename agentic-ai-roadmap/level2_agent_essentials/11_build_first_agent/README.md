# 11 · Build Your First Agent (from scratch)

> **Goal:** Write the agent loop yourself so nothing about agents is magic.

## The loop in ~20 lines
```python
messages = [{"role": "user", "content": goal}]
for step in range(MAX_STEPS):
    response = llm(system, messages, tools)
    messages.append({"role": "assistant", "content": response.content})   # keep it verbatim
    if response.stop_reason != "tool_use":
        return final_text(response)                                       # done
    results = [run(tool_call) for tool_call in tool_uses(response)]       # act
    messages.append({"role": "user", "content": results})                 # observe
```

## Rules that prevent 90 % of bugs
- Append the **whole** `response.content` (it can contain thinking + text + several tool calls).
- Every `tool_use` block needs a `tool_result` with the **same `tool_use_id`**, all in **one** user message.
- Send tool errors back with `is_error: true` instead of crashing — the agent will often fix its own mistake.
- Always cap `MAX_STEPS` (and ideally tokens/cost).
- Log every step. You can't debug what you can't see.

## The agent you'll build
A **Data Analyst Agent** for the wellness-tourism dataset in `data/`. Tools (from `common/tools.py`):
`dataset_columns`, `dataset_stats`, `conversion_rate_by`, `calculator`.

## Hands-on
```bash
python level2_agent_essentials/11_build_first_agent/data_analyst_agent.py
python level2_agent_essentials/11_build_first_agent/data_analyst_agent.py "Which customer segment should sales target first, and why?"
```
Watch the trace: the agent discovers the columns, chooses analyses, does maths with the calculator and writes a recommendation — none of which you scripted.

## Exercises
1. Add `search_repo` and `read_repo_file` to the tool list and ask "Does the training code handle class imbalance?"
2. Set `MAX_STEPS = 2`. What happens? How should an agent behave when it runs out of steps?
3. Track total input/output tokens across the run and print an estimated cost.
