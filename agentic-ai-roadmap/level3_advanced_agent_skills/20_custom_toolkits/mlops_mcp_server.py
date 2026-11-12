"""Lesson 20 – an MCP server exposing this repo's tools as a reusable toolkit.

Run:  pip install "mcp[cli]>=2"
      python level3_advanced_agent_skills/20_custom_toolkits/mlops_mcp_server.py
(It speaks MCP over stdio, so it just waits for a client to connect.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer  # mcp 2.x (was FastMCP in 1.x)

from common import tools as T

mcp = MCPServer("mlops-toolkit")

# The plain functions from common/tools.py become MCP tools as-is: MCPServer reads
# their type hints and docstrings to build the schemas, exactly like @beta_tool.
for fn in (T.dataset_columns, T.dataset_stats, T.conversion_rate_by, T.calculator,
           T.search_repo, T.list_repo_files, T.read_repo_file):
    mcp.tool()(fn)


@mcp.resource("dataset://schema")
def dataset_schema() -> str:
    """Column names and dtypes of the wellness-tourism dataset."""
    return T.dataset_columns()


@mcp.prompt()
def weekly_report(focus: str = "conversion") -> str:
    """Prompt template for a weekly sales-analytics report."""
    return (f"Using the mlops-toolkit tools, write a weekly report focused on {focus}: "
            "3 key numbers, 2 insights, 1 recommended action.")


if __name__ == "__main__":
    mcp.run()  # stdio transport by default; use mcp.run(transport="streamable-http") to serve over HTTP
