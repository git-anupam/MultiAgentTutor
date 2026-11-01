"""
core/graph.py - builds the LangGraph graph that runs the agents.

BACKGROUND: WHAT IS THE GRAPH?
------------------------------
LangGraph describes a program as a flowchart:

    * NODES are steps (each one a Python function, e.g. an agent's turn)
    * EDGES are arrows saying which step runs next
    * START and END are built-in markers for "begin here" and "stop here"

Each time the user sends a message, the graph runs once from START to END -
or until it PAUSES to ask the user something.

THE COMPLETE GRAPH (step 7)
---------------------------
                                  +------> END   (reply finished)
                                  |
    START -> router ---> tutor ---+------> quiz_maker / tutor   (handoff:
                    \      ^      |                 next_agent or return_stack)
                     \     |      +------> ask_user  (clarifying question)
                      \    |                  |
                       --> quiz_maker ---(same three choices)
                           |                  |
                           +<-- Command(goto=<agent that asked>) after resume

    router      picks the first agent for each new message (core/router.py)
    <agents>    one node per entry in agents/registry.py, each wrapped with
                the handoff bookkeeping (core/handoff.py: agent_node)
    ask_user    pauses for a clarifying question (core/clarify.py)

THREE KINDS OF ARROW
--------------------
    router   -> agent      the router RETURNS Command(goto=<agent>)
    agent    -> ?          a CONDITIONAL edge: route_after_agent()
                           (core/handoff.py) reads the state and picks
                           ask_user, another agent, or END
    ask_user -> agent      the node RETURNS Command(goto=<agent that asked>)

Nothing in this file names a specific agent: everything comes from the
registry. Adding Agent 3 does not change the graph code.
"""

from __future__ import annotations

from langgraph.graph import START, StateGraph

from agents.registry import AGENTS
from core.clarify import ASK_USER, ask_user_node
from core.handoff import agent_node, route_after_agent
from core.logging_setup import get_logger
from core.router import ROUTER, router_node
from core.state import ChatState

log = get_logger("graph")


def build_graph(checkpointer):
    """
    Build and compile the agent graph.

    Args:
        checkpointer: where to save conversation state between runs (the
            Postgres checkpointer from infra/postgres.py). Required: pausing
            for a clarifying question means saving the half-finished run
            somewhere, and the checkpointer is that somewhere.

    Returns:
        A compiled graph. Run it with
            graph.invoke(input, {"configurable": {"thread_id": ...}})
    """
    # A StateGraph is told the shape of its state up front, so it knows which
    # keys exist and which reducer (e.g. add_messages) applies to each.
    builder = StateGraph(ChatState)

    # --- The router: first stop for every new message ----------------------
    # No edges are added out of it: it returns Command(goto=<agent>).
    builder.add_node(ROUTER, router_node)
    builder.add_edge(START, ROUTER)

    # --- One node per registered agent -------------------------------------
    # This loop is why adding an agent never requires editing this file.
    for spec in AGENTS.values():
        # agent_node() wraps the agent's own function with the handoff
        # bookkeeping (core/handoff.py).
        builder.add_node(spec.name, agent_node(spec))
        # After any agent: ask_user, another agent, or END.
        builder.add_conditional_edges(spec.name, route_after_agent)

    # --- The clarifying-question node --------------------------------------
    # No edges out of it either: it returns Command(goto=<agent that asked>).
    builder.add_node(ASK_USER, ask_user_node)

    # "Compiling" checks the graph for mistakes (like an arrow pointing at a
    # node that does not exist) and turns it into a runnable object.
    graph = builder.compile(checkpointer=checkpointer)

    log.info(
        "graph built: %s -> [%s], plus %s (checkpointer: %s)",
        ROUTER, ", ".join(AGENTS), ASK_USER, type(checkpointer).__name__,
    )
    return graph
