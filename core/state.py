"""
core/state.py - the shape of the data that flows through the agent graph.

BACKGROUND: WHAT IS "STATE" IN LANGGRAPH?
-----------------------------------------
LangGraph runs a graph of steps ("nodes"). Each node is a plain Python
function. Nodes do not talk to each other directly; instead they all read from
and write to one shared dictionary called the STATE.

    state  ->  node A  ->  updated state  ->  node B  ->  updated state ...

A node receives the current state and returns only the keys it wants to
change. LangGraph merges those changes into the state and passes it on.

The state is also exactly what the checkpointer saves to Postgres in step 3.
So "what does the program remember between messages?" has a precise answer:
whatever is in this TypedDict.

FIELDS SO FAR
-------------
    messages            the conversation (step 2)
    pending_question    a clarifying question waiting for an answer (step 5)
    quiz                the most recent quiz from the Quiz Maker (step 6)
    next_agent          handoff bookkeeping: who runs next (step 7)
    return_stack        handoff bookkeeping: who is waiting (step 7)
    handoffs_this_turn  handoff bookkeeping: loop protection (step 7)
    task                the job description passed with a handoff (step 7)

Every field after `messages` is read with `state.get(...)`, because
conversations saved by earlier build steps do not have it.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """Everything the graph knows about one conversation."""

    # The full conversation: HumanMessage (user), AIMessage (an agent), and
    # later ToolMessage (a tool's result).
    #
    # THE `Annotated[..., add_messages]` PART IS IMPORTANT.
    # By default, when a node returns {"messages": [...]}, LangGraph would
    # REPLACE the old value with the new one - the history would be lost.
    # `add_messages` is a "reducer": a small function that tells LangGraph how
    # to combine the old value with the new one. This one APPENDS new messages
    # to the list instead. So a node can return just its single reply:
    #
    #     return {"messages": [reply]}
    #
    # and the history grows by one message.
    messages: Annotated[list[AnyMessage], add_messages]

    # Set when an agent needs more information before it can answer, e.g.
    #     {"agent": "tutor", "question": "Which topic should I explain?"}
    # and cleared (back to None) once the user has answered. While it is set,
    # the graph routes to the ask_user node, which pauses. See core/clarify.py.
    #
    # No reducer here, so a node returning {"pending_question": ...} simply
    # REPLACES the old value - which is what we want for a single slot.
    #
    # Nodes read it with `state.get("pending_question")`, not
    # `state["pending_question"]`, because conversations saved before step 5
    # do not have this key at all.
    pending_question: dict | None

    # The latest quiz written by the Quiz Maker, as plain data:
    #     {"topic": "gravity",
    #      "questions": [{"question": "...", "choices": ["...", ...],
    #                     "answer": "B", "explanation": "..."}, ...]}
    # Kept here, NOT in `messages`, so the answers are never shown in the
    # chat; the Tutor reads it to present the quiz and grade (step 7).
    # The Tutor adds "presented": True once it has shown the questions.
    quiz: dict | None

    # --- Handoff bookkeeping (step 7, see core/handoff.py) -----------------
    # Only the wrapper in core/handoff.py and the router write these; agent
    # code never touches them directly.

    # The agent a handoff named to run next, e.g. "quiz_maker". Cleared as
    # soon as that agent starts.
    next_agent: str | None

    # Agents waiting for control to come back after a "call" handoff, most
    # recent last. ["tutor"] = "the Tutor called someone and is waiting".
    return_stack: list[str]

    # Handoffs so far while answering the current message. Stops two agents
    # from passing work back and forth forever. Reset by the router.
    handoffs_this_turn: int

    # The job description passed along with the latest handoff, e.g.
    # "3-question quiz on gravity" from "HANDOFF: quiz_maker | 3-question
    # quiz on gravity". The receiving agent reads it. Reset by the router.
    task: str | None
