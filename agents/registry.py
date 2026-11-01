"""
agents/registry.py - the ONE place that lists every agent.

Everything that needs to know "which agents exist" reads this dictionary:

    core/graph.py   adds one graph node per agent, and wires each one to the
                    clarifying-question and handoff flows
    core/router.py  reads each agent's `description` to decide which agent
                    handles the user's message
    agents/base.py  tells agents with `can_hand_off=True` which other agents
                    they can hand work to

ADDING AGENT 3
--------------
    1. agents/<name>.py           the agent file, ending in `AGENT = AgentSpec(...)`
    2. agents/prompts/<name>.md   its system prompt
    3. one line in `_ALL` below

Nothing else changes: not the graph, not the router, not checkpointing, not
tracing.
"""

from __future__ import annotations

from agents import quiz_maker, tutor
from agents.base import AgentSpec

# The order does not matter for routing; it is just the order agents are
# listed in logs and help text.
_ALL: list[AgentSpec] = [
    tutor.AGENT,
    quiz_maker.AGENT,
]

# {"tutor": <AgentSpec>, "quiz_maker": <AgentSpec>}
AGENTS: dict[str, AgentSpec] = {spec.name: spec for spec in _ALL}

# The agent that handles a message when nothing more specific applies.
DEFAULT_AGENT = tutor.AGENT.name
