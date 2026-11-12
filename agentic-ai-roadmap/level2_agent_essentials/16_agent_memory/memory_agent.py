"""Lesson 16 – short-term, summary and long-term memory.

Run:  python level2_agent_essentials/16_agent_memory/memory_agent.py [--demo]
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anthropic import beta_tool

from common import MODEL, ask, banner, client, fallback_kwargs, require_api_key, text_of
from common.tools import conversion_rate_by, dataset_stats, runner_tools

STORE = Path(__file__).with_name("memory_store.json")
SUMMARY_AFTER_MESSAGES = 12  # compress short-term memory beyond this many messages


# ---------------------------------------------------------------- long-term memory (a JSON file)
def _load() -> dict:
    return json.loads(STORE.read_text()) if STORE.exists() else {}


@beta_tool
def save_memory(key: str, value: str) -> str:
    """Save a durable fact about the user or project for future sessions (preferences, goals, decisions).

    Args:
        key: Short snake_case identifier, e.g. "preferred_format" or "target_segment".
        value: The fact to remember.
    """
    data = _load()
    data[key] = {"value": value, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    STORE.write_text(json.dumps(data, indent=2))
    return f"Saved memory '{key}'."


@beta_tool
def search_memory(query: str) -> str:
    """Search long-term memory for facts saved in earlier sessions. Use at the start of a task.

    Args:
        query: Words to look for (matches keys and values). Use "*" to list everything.
    """
    data = _load()
    words = query.lower().split()
    hits = {k: v["value"] for k, v in data.items()
            if query == "*" or any(w in f"{k} {v['value']}".lower() for w in words)}
    return json.dumps(hits) if hits else "No matching memories."


SYSTEM = """You are a helpful analytics assistant for a wellness-tourism sales team.
Memory: at the start of each new topic, call search_memory to recall relevant facts.
When the user shares a lasting preference, goal or decision, call save_memory.
You can also analyse the customer dataset with the data tools."""


# ---------------------------------------------------------------- short-term memory with summarisation
class Conversation:
    def __init__(self):
        self.summary = ""
        self.messages: list[dict] = []

    def _compress(self):
        """Summary memory: fold the oldest turns into a running summary."""
        if len(self.messages) <= SUMMARY_AFTER_MESSAGES:
            return
        old, self.messages = self.messages[:-4], self.messages[-4:]
        while self.messages and self.messages[0]["role"] != "user":  # history must start with a user turn
            old.append(self.messages.pop(0))
        transcript = "\n".join(f"{m['role']}: {m['content'] if isinstance(m['content'], str) else '[tool activity]'}" for m in old)
        self.summary = ask(f"Previous summary: {self.summary or 'none'}\n\nNew turns:\n{transcript}\n\n"
                           "Update the summary in <=120 words, keeping facts, decisions and open questions.",
                           max_tokens=1000, output_config={"effort": "low"})
        print(f"   🧠 compressed {len(old)} old messages into summary memory")

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        system = SYSTEM + (f"\n\n<conversation_summary>{self.summary}</conversation_summary>" if self.summary else "")
        runner = client.beta.messages.tool_runner(
            model=MODEL, max_tokens=16000, system=system, max_iterations=8,
            tools=[save_memory, search_memory, *runner_tools(dataset_stats, conversion_rate_by)],
            messages=self.messages, **fallback_kwargs(),
        )
        final = None
        for message in runner:
            for b in message.content:
                if b.type == "tool_use":
                    print(f"   🔧 {b.name}({b.input})")
            final = message
        answer = text_of(final)
        # Keep only the text in short-term memory (tool chatter is not needed later).
        self.messages.append({"role": "assistant", "content": answer or "(no answer)"})
        self._compress()
        return answer


def demo():
    STORE.unlink(missing_ok=True)
    banner("SESSION 1")
    s1 = Conversation()
    for msg in ["Hi! I'm Arun from the Bangalore sales team. Always give me answers as 3 short bullets.",
                "Our quarterly goal is to grow Deluxe package sales. Remember that."]:
        print(f"\n👤 {msg}\n🤖 {s1.send(msg)}")

    banner("SESSION 2 – brand-new conversation, nothing in `messages`")
    s2 = Conversation()
    msg = "Which customer occupation should I focus on this quarter?"
    print(f"\n👤 {msg}\n🤖 {s2.send(msg)}")
    print(f"\nLong-term memory file ({STORE.name}):\n{STORE.read_text()}")


if __name__ == "__main__":
    require_api_key()
    if "--demo" in sys.argv:
        demo()
    else:
        conv = Conversation()
        print("Chat with the memory agent (Ctrl+C to quit). Memories persist in", STORE)
        while True:
            print("🤖", conv.send(input("\n👤 ")))
