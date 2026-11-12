# 24 · Deploy to Production (host your agent for real users)

> **Goal:** Ship an agent as a secure, observable, scalable service — using the same MLOps habits this repo already has.

## Production architecture
```
 users / apps ──HTTPS──► API gateway / auth ──► Agent service (FastAPI, N replicas, stateless)
                                                   │        │           │
                                                   │        │           └─► logs, traces, metrics (Datadog / OTel)
                                                   │        └─► tools: DBs, internal APIs, MCP servers
                                                   └─► Claude API (retries, timeouts, fallbacks)
                         state (conversations, memory, task queues) ─► Redis / Postgres — not in process memory
```

## Production checklist

| Area | What `app.py` / `Dockerfile` do | What you'd add at scale |
|---|---|---|
| **Secrets** | `ANTHROPIC_API_KEY`, `SERVICE_API_KEY` from env | Secret manager (Vault, AWS SM, GCP SM) |
| **AuthN / rate limits** | API-key header + per-key sliding-window limit | Gateway auth (OAuth/JWT), Redis-backed limits |
| **Least privilege** | Only read-only tools exposed | Per-user tool permissions, approval queues for writes |
| **Reliability** | Timeouts, SDK retries, refusal fallbacks, max iterations, mapped HTTP errors | Circuit breakers, queue long jobs (Celery/SQS), idempotency keys |
| **Observability** | Structured JSON logs: request id, tools, tokens, latency, stop reason | Traces per agent step (OpenTelemetry), dashboards, cost alerts |
| **Quality gates** | — | Run the lesson-18 eval in CI; block deploy if pass rate drops |
| **Containers** | Slim image, non-root user, healthcheck | Image scanning, SBOM, resource limits |
| **Scaling** | Stateless service | Horizontal autoscaling; streaming responses (SSE) for long answers |
| **Safety** | System prompt treats tool output as data | Lesson-19 guards as middleware, PII redaction in logs |

## Hosting options
| Option | Good for |
|---|---|
| **Hugging Face Spaces (Docker)** | Demos — the same path `hosting/hosting.py` already uses for the Streamlit app |
| **Google Cloud Run / AWS App Runner / Azure Container Apps** | Serverless containers, scale to zero |
| **Kubernetes (EKS/GKE/AKS)** | Many services, custom networking, GPUs for local models |
| **Claude Managed Agents** | Let Anthropic host the agent loop + sandbox; you call it via API (no container to run) |

## Hands-on
```bash
# 1) Run locally
pip install -r level3_advanced_agent_skills/24_deploy_to_production/requirements.txt
SERVICE_API_KEY=dev-secret python level3_advanced_agent_skills/24_deploy_to_production/app.py
curl -s -X POST localhost:8000/ask -H "X-API-Key: dev-secret" -H "Content-Type: application/json" \
     -d '{"question": "Which occupation converts best, and which file trains the model?"}'

# 2) Run in Docker (build from the repo root)
cd ..   # repo root
docker build -f agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production/Dockerfile -t mlops-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -e SERVICE_API_KEY=change-me mlops-agent
```

### CI/CD sketch (extend `.github/workflows/pipeline.yml`)
```yaml
  agent-eval:                       # quality gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r agentic-ai-roadmap/requirements.txt
      - run: python agentic-ai-roadmap/level2_agent_essentials/18_agent_evaluation/eval_harness.py
        env: {ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}"}
  deploy-agent:
    needs: agent-eval
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production/Dockerfile -t mlops-agent .
      # push to your registry and roll out (Cloud Run / HF Spaces / k8s)
```

## Exercises
1. Add a `/ask/stream` endpoint using Server-Sent Events.
2. Move rate-limit state into Redis so it works across replicas.
3. Add OpenTelemetry tracing with one span per agent step and tool call.
4. Make `eval_harness.py` exit non-zero when pass rate < 0.8 so CI actually blocks the deploy.
