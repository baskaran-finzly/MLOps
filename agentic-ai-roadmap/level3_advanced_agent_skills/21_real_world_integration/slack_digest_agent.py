"""Lesson 21 – an agent that analyses data and posts a digest to Slack.

Run:  python level3_advanced_agent_skills/21_real_world_integration/slack_digest_agent.py [--send]
Posting is a dry run unless --send is given AND SLACK_WEBHOOK_URL is set.
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anthropic import beta_tool

from common import MODEL, banner, client, fallback_kwargs, require_api_key, text_of
from common.tools import calculator, conversion_rate_by, dataset_stats, runner_tools

SEND = "--send" in sys.argv
WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
_posted: set[str] = set()  # idempotency: don't post the same message twice if the agent retries


@beta_tool
def post_to_slack(text: str) -> str:
    """Post a message to the team's #sales-analytics Slack channel. Use Slack mrkdwn (*bold*, • bullets).

    Call this exactly once, with the final digest.

    Args:
        text: The full message to post (max ~3000 characters).
    """
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _posted:
        return "Already posted this exact message; not posting again."
    _posted.add(key)
    if not (SEND and WEBHOOK):
        print("\n----- DRY RUN: would post to Slack -----\n" + text + "\n----------------------------------------")
        return "Dry run: message rendered but not sent (run with --send and SLACK_WEBHOOK_URL to post)."
    req = urllib.request.Request(WEBHOOK, data=json.dumps({"text": text[:3000]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return f"Posted to Slack (HTTP {resp.status})."


SYSTEM = """You produce a short daily analytics digest for the sales team's Slack channel.
1. Use the data tools to get: overall conversion rate, best and worst ProductPitched, best CityTier, and one surprising insight.
2. Use the calculator for any arithmetic.
3. Call post_to_slack ONCE with a digest: a bold title, 4-6 bullets with numbers, and one recommended action.
Keep it under 120 words. Only use numbers from tools."""

if __name__ == "__main__":
    require_api_key()
    banner(f"Slack digest agent  (mode: {'SEND' if SEND and WEBHOOK else 'dry run'})")
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000, system=SYSTEM, max_iterations=12,
        tools=[*runner_tools(dataset_stats, conversion_rate_by, calculator), post_to_slack],
        messages=[{"role": "user", "content": "Create and post today's digest."}],
        **fallback_kwargs(),
    )
    final = None
    for message in runner:
        for b in message.content:
            if b.type == "tool_use" and b.name != "post_to_slack":
                print(f"🔧 {b.name}({b.input})")
        final = message
    print("\nAgent:", text_of(final))
