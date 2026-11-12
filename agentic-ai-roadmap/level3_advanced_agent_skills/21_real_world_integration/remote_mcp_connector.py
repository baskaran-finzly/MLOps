"""Lesson 21 – use a hosted MCP server (Notion, Linear, Asana, …) via the API's MCP connector.

Anthropic's API connects to the remote MCP server for you: no local MCP client code.

Run:  export REMOTE_MCP_URL=https://mcp.notion.com/mcp
      export REMOTE_MCP_TOKEN=<OAuth access token for that service>
      python level3_advanced_agent_skills/21_real_world_integration/remote_mcp_connector.py "your request"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import create, require_api_key, text_of

require_api_key()
url, token = os.getenv("REMOTE_MCP_URL"), os.getenv("REMOTE_MCP_TOKEN")
if not url:
    sys.exit("Set REMOTE_MCP_URL (and REMOTE_MCP_TOKEN if the server needs auth). See README.")

server = {"type": "url", "url": url, "name": "remote"}
if token:
    server["authorization_token"] = token

request = " ".join(sys.argv[1:]) or "What tools do you have access to? Then use one read-only tool to show me something useful."
response = create(
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[server],
    # Every server needs a matching toolset. Allow-list mode: everything disabled except read tools you name, e.g.
    # {"type": "mcp_toolset", "mcp_server_name": "remote", "default_config": {"enabled": False},
    #  "configs": {"search": {"enabled": True}}}
    tools=[{"type": "mcp_toolset", "mcp_server_name": "remote"}],
    system="Prefer read-only actions. Ask before creating, editing or deleting anything.",
    messages=[{"role": "user", "content": request}],
)

for block in response.content:
    if block.type == "mcp_tool_use":
        print(f"🔧 remote tool: {block.name}({block.input})")
print("\n" + text_of(response))
