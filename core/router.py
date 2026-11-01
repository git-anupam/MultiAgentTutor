"""
core/router.py - picks which agent handles each user message.

BACKGROUND: WHAT IS A ROUTER?
-----------------------------
With more than one agent, something has to decide who answers. The ROUTER is
the first node of every run:

    START -> router -> tutor        "what is gravity?"
                    -> tutor        "1-B, 2-A, 3-C"   (answers to a quiz)
                    -> quiz_maker   "make it harder"  (changing the quiz)

It runs again for EVERY new message, so the conversation can switch agents
at any point ("mid-conversation switches").

HOW IT DECIDES
--------------
It asks the model. The prompt (agents/prompts/router.md) lists every agent
in agents/registry.py with its one-line `description`, shows the last few
messages, and asks for a JSON answer:

    {"reason": "student wants a harder version of the quiz", "agent": "quiz_maker"}

Ollama's STRUCTURED OUTPUT (`format=<JSON schema>`) forces the reply into
exactly that shape, and the schema's `enum` only allows real agent names, so
the router cannot invent an agent. If anything still goes wrong, it falls
back to the default agent (the Tutor).

The router reads the agent list from the registry at run time, so adding an
agent never requires editing this file - only writing a good `description`.

WHAT IT DOES NOT DO
-------------------
It does not run when the user is ANSWERING a clarifying question: that
input resumes the paused run at the ask_user node (core/clarify.py), which
already knows which agent asked.
"""

from __future__ import annotations

import json

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from mlflow.entities import SpanType

from agents.registry import AGENTS, DEFAULT_AGENT
from core.llm import build_chat_model
from core.logging_setup import get_logger
from core.prompts import load_prompt
from core.state import ChatState

log = get_logger("router")

ROUTER = "router"

# How many recent messages the router looks at. Enough to see "the Tutor
# just gave a quiz", small enough to keep this extra model call quick.
RECENT_MESSAGES = 6


def _describe_agents() -> str:
    """The agent list for the prompt, built from the registry."""
    return "\n".join(f"- {name}: {spec.description}" for name, spec in AGENTS.items())


def _recent_conversation(state: ChatState) -> str:
    """
    The few messages BEFORE the latest one, as plain text, each shortened if
    very long. The latest message is shown to the model separately (see
    router_node), so it cannot be confused with older requests.
    """
    lines = []
    for message in state["messages"][-RECENT_MESSAGES - 1:-1]:
        who = "Student" if message.type == "human" else (message.name or "assistant")
        text = message.content.replace("\n", " ")
        lines.append(f"{who}: {text[:300]}")
    return "\n".join(lines)


def router_node(state: ChatState) -> Command:
    """
    Choose an agent for the latest user message and go there.

    Returns a `Command`: the state changes to make AND the node to run next,
    in one object. The changes reset the per-message handoff bookkeeping
    (see core/handoff.py), because a new message starts a fresh run.
    """
    names = list(AGENTS)
    fresh_turn = {"next_agent": None, "return_stack": [], "handoffs_this_turn": 0, "task": None}

    # Nothing to decide with a single agent - skip the model call.
    if len(names) == 1:
        return Command(goto=names[0], update=fresh_turn)

    prompt = (
        load_prompt("router.md")
        .replace("{agents}", _describe_agents())
        .replace("{default}", DEFAULT_AGENT)
    )

    # The JSON schema the reply must follow. `reason` comes first so the
    # model "thinks" a little before naming the agent.
    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "agent": {"type": "string", "enum": names},
        },
        "required": ["reason", "agent"],
    }
    # temperature 0: routing should be consistent, not creative.
    # num_predict caps the reply length; the JSON is short.
    model = build_chat_model(temperature=0.0, format=schema, num_predict=100)

    with mlflow.start_span(name="route", span_type=SpanType.CHAIN) as span:
        earlier = _recent_conversation(state) or "(none)"
        latest = state["messages"][-1].content
        span.set_inputs({"earlier_messages": earlier, "latest_message": latest, "agents": names})

        # The latest message is set apart and named explicitly. Small models
        # otherwise latch on to an older request in the history (e.g. treat
        # quiz answers as another "make it harder").
        reply = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=(
                f"Earlier messages, for context only:\n{earlier}\n\n"
                f"THE STUDENT'S LATEST MESSAGE:\n\"{latest}\"\n\n"
                "Which agent should handle the latest message?"
            )),
        ])

        try:
            decision = json.loads(reply.content)
            agent, reason = decision["agent"], decision.get("reason", "")
            if agent not in AGENTS:
                raise ValueError(f"unknown agent {agent!r}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            log.warning("could not read the router's answer (%s); using %s", error, DEFAULT_AGENT)
            agent, reason = DEFAULT_AGENT, f"fallback: {error}"

        span.set_attributes({"chosen_agent": agent})
        span.set_outputs({"agent": agent, "reason": reason})

    log.info("-> %s (%s)", agent, reason)
    return Command(goto=agent, update=fresh_turn)
