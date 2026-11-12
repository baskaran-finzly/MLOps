# 21 · Real-World Integration (Slack, Notion, Gmail…)

> **Goal:** Connect agents to the tools your team actually uses — safely.

## Three ways to integrate

| Approach | How | Pros | Cons |
|---|---|---|---|
| **A. Direct API tool** | Write a tool that calls the service's REST API / SDK (`slack_sdk`, `notion-client`, Gmail API) | Full control, easy to gate | You maintain auth + code per service |
| **B. Local MCP server** | Run an existing MCP server (Slack, Notion, Gmail, GitHub…) and connect like lesson 20 | Reuse community servers | Runs on your machine/infra |
| **C. Remote MCP connector** | Pass `mcp_servers=[{url, authorization_token}]` in the API request; Anthropic connects server-side | Zero local code | Needs a hosted MCP server + OAuth token |

## Production rules for integrations
- **Least privilege** – a bot token that can post to *one* channel, not admin scopes.
- **Secrets in env vars / a vault**, never in prompts or code.
- **Read freely, write carefully** – reading Notion is low-risk; sending email or posting publicly needs **approval** or a **dry-run** default.
- **Idempotency** – don't double-post when the agent retries; use dedupe keys.
- **Rate limits & retries** – respect each API's limits (Slack: ~1 msg/sec/channel).
- **Untrusted content** – emails and pages can contain prompt injections (lesson 19).

## Hands-on

### A. Daily analytics digest → Slack (direct API tool)
```bash
# Dry run (prints what it would post)
python level3_advanced_agent_skills/21_real_world_integration/slack_digest_agent.py

# Real post: create a Slack Incoming Webhook (api.slack.com/messaging/webhooks)
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
python level3_advanced_agent_skills/21_real_world_integration/slack_digest_agent.py --send
```
The agent analyses the dataset, drafts a digest, and calls `post_to_slack`. Posting is **dry-run unless `--send`** is passed.

### C. Remote MCP connector (Notion, Linear, Asana, … any hosted MCP server)
```bash
export REMOTE_MCP_URL=https://mcp.notion.com/mcp        # the service's MCP endpoint
export REMOTE_MCP_TOKEN=...                              # OAuth access token for that service
python level3_advanced_agent_skills/21_real_world_integration/remote_mcp_connector.py "List my 5 most recently edited pages"
```

## Exercises
1. Swap the webhook for `slack_sdk.WebClient.chat_postMessage` with a bot token, and add threaded replies.
2. Add a `create_notion_page` tool that saves the digest to a Notion database.
3. Build a Gmail triage agent (read-only first!) that labels unread emails as *urgent / FYI / spam*.
