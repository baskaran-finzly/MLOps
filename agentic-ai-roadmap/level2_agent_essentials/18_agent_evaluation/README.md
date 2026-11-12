# 18 · Agent Evaluation (accuracy, performance, reliability)

> **Goal:** Prove your agent works — and know immediately when a change makes it worse.

You'd never ship the XGBoost model in this repo without a test set and ROC-AUC. Agents need the same discipline; they're just harder to score.

## What to measure

| Dimension | Metrics | How |
|---|---|---|
| **Task success** | pass rate, exact/numeric match | Code-based graders vs. ground truth |
| **Answer quality** | correctness, faithfulness (no invented numbers), helpfulness | **LLM-as-judge** with a rubric + structured output |
| **Trajectory** | tools chosen, # steps, errors, loops | Inspect the transcript; assert required tools were used |
| **Efficiency** | tokens, cost, latency | `response.usage`, timers |
| **Reliability** | pass^k (passes *k* times in a row), variance | Run each case several times |
| **Safety** | refused harmful/out-of-scope asks, no data leaks | Adversarial test cases (lesson 19) |

## Grader types (prefer the top ones)
1. **Code-based** – exact match, numeric tolerance, regex, JSON schema, unit tests. Fast, cheap, objective.
2. **LLM-as-judge** – rubric-based scoring for open-ended answers. Give the judge the ground truth/evidence; ask for reasoning + a verdict in a schema; spot-check it against humans.
3. **Human review** – gold standard; use to calibrate the others.

## Good eval sets
- Start with **20-50 real cases**, including edge cases and "should refuse / doesn't know" cases.
- Derive ground truth **from the source of truth** (here: pandas on the CSV), not from the model.
- Keep a held-out set; re-run on every prompt/model/tool change (regression testing in CI — like `pipeline.yml`).
- Track results over time — this script can log to **MLflow**, the same tracker the repo already uses.

## Hands-on
```bash
python level2_agent_essentials/18_agent_evaluation/eval_harness.py               # run all cases
python level2_agent_essentials/18_agent_evaluation/eval_harness.py --repeats 3   # reliability (pass^k)
python level2_agent_essentials/18_agent_evaluation/eval_harness.py --mlflow      # log metrics to MLflow
```
It evaluates the lesson-11 **Data Analyst Agent** with code graders + an LLM judge, prints a scorecard and saves `eval_results.json`.

## Exercises
1. Add 5 more cases, including one where the right answer is "that column doesn't exist".
2. Change the agent's system prompt to something worse — does the eval catch the regression?
3. Add a trajectory check: fail any run that used more than 8 tool calls.
4. Wire the eval into a GitHub Actions job that fails the build if pass rate < 80 %.
