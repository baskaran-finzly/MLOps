# Build from the REPO ROOT so the image contains the dataset and code the agent's tools read:
#   docker build -f agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production/Dockerfile -t mlops-agent .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -e SERVICE_API_KEY=change-me mlops-agent
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /srv/MLOps

# Dependencies first for better layer caching
COPY agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Only what the service needs: shared agent code + the repo files the tools read
COPY agentic-ai-roadmap/common agentic-ai-roadmap/common
COPY agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production
COPY data data
COPY model_building model_building
COPY deployment deployment
COPY hosting hosting
COPY .github .github
COPY README.md README.md

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /srv/MLOps
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["python", "agentic-ai-roadmap/level3_advanced_agent_skills/24_deploy_to_production/app.py"]
