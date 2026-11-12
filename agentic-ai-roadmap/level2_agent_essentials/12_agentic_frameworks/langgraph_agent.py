"""Lesson 12 (optional) – the same agent in LangGraph.

Run:  pip install langgraph langchain-anthropic
      python level2_agent_essentials/12_agentic_frameworks/langgraph_agent.py

LangGraph models an agent as a graph: an "agent" node (LLM) and a "tools"
node, with an edge that loops back until the LLM stops calling tools.
`create_react_agent` builds exactly that graph for you.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common import MODEL, require_api_key
from common import tools as T

require_api_key()

# Re-use the same plain functions; LangChain's @tool reads their signatures and docstrings.
lc_tools = [tool(T.dataset_columns), tool(T.dataset_stats), tool(T.conversion_rate_by), tool(T.calculator)]

agent = create_react_agent(
    ChatAnthropic(model=MODEL, max_tokens=16000),
    lc_tools,
    prompt="You are a data analyst for a wellness-tourism company. Use tools; never invent numbers.",
)

goal = " ".join(sys.argv[1:]) or "Which CityTier converts best, and how many customers are in it?"
result = agent.invoke({"messages": [("user", goal)]}, {"recursion_limit": 25})

for msg in result["messages"]:
    kind = type(msg).__name__
    if getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            print(f"🔧 {call['name']}({call['args']})")
    elif kind == "AIMessage":
        print(f"\nANSWER:\n{msg.content if isinstance(msg.content, str) else msg.text()}")
