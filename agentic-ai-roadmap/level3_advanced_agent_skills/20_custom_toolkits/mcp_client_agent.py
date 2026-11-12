"""Lesson 20 – an agent that discovers and uses tools from an MCP server.

Run:  pip install "mcp[cli]>=2" "anthropic[mcp]"
      python level3_advanced_agent_skills/20_custom_toolkits/mcp_client_agent.py ["question"]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from common import MODEL, fallback_kwargs, require_api_key, text_of

SERVER = Path(__file__).with_name("mlops_mcp_server.py")


async def main(question: str):
    client = AsyncAnthropic()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            print("Tools discovered from MCP server:", ", ".join(t.name for t in listed.tools))

            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=16000,
                max_iterations=10,
                system="You are an MLOps analyst. Use the toolkit to answer with evidence and file citations.",
                tools=[async_mcp_tool(t, session) for t in listed.tools],
                messages=[{"role": "user", "content": question}],
                **fallback_kwargs(),
            )
            final = None
            async for message in runner:
                for b in message.content:
                    if b.type == "tool_use":
                        print(f"🔧 {b.name}({b.input})")
                final = message
            print("\n" + text_of(final))


if __name__ == "__main__":
    require_api_key()
    q = " ".join(sys.argv[1:]) or "Which CityTier converts best, and which file trains the model?"
    asyncio.run(main(q))
