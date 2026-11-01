"""
agents/base.py - what every agent has in common.

BACKGROUND: WHAT IS AN AGENT HERE?
----------------------------------
In this project an "agent" is these things bundled together:

    * a SYSTEM PROMPT  - its standing instructions (a Markdown file)
    * a MODEL CALL     - the conversation is sent to the local LLM
    * a NODE FUNCTION  - the Python function LangGraph runs when it is this
                         agent's turn
    * TOOLS (optional) - Python functions the model may ask us to run

This file holds the pieces every agent shares, so the individual agent files
stay short:

    AgentSpec              a small record describing one agent
    call_model()           sends the conversation to the model with the
                           agent's prompt, and runs the TOOL-CALLING LOOP
                           if the agent has tools (step 6)
    run_tool()             runs one tool the model asked for
    clarification_update() turns a "CLARIFY: ..." reply into a pause (step 5)
    run_agent()            one full turn for a simple agent: call the model,
                           then reply, ask a clarifying question, or hand
                           off to another agent (step 7)

WHY A SPEC OBJECT?
------------------
`agents/registry.py` is a dictionary of every agent's spec. The graph builder
reads each agent's `name` and `node` to wire it in, and the router reads each
`description` to decide who should answer. Giving every agent the
same `AgentSpec` shape is what makes "add Agent 3 = one new file, one new
prompt, one registry entry" possible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import mlflow
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from mlflow.entities import SpanType

from core.clarify import parse_clarification
from core.handoff import parse_handoff
from core.llm import build_chat_model
from core.logging_setup import get_logger
from core.prompts import load_prompt

# Safety limit for the tool-calling loop. A confused model could keep asking
# for tools forever; after this many rounds we make it answer without tools.
MAX_TOOL_ROUNDS = 3


@dataclass(frozen=True)
class AgentSpec:
    """A description of one agent. Every agent file creates exactly one."""

    # Short unique id, e.g. "tutor". Also used as the agent's node name in the
    # LangGraph graph and as the `name` stamped on its messages, so you can
    # always tell which agent said what.
    name: str

    # One plain-English sentence about what the agent is for. The router
    # (core/router.py) reads it to decide which agent handles a message, so
    # write it for the router: say when to pick this agent, and when not to.
    description: str

    # File name inside agents/prompts/, e.g. "tutor.md".
    prompt_file: str

    # The function LangGraph runs for this agent. It receives the state and
    # returns the changes to make to it.
    node: Callable[[dict], dict]

    # Optional per-agent creativity setting. `None` means "use the default
    # from .env".
    temperature: float | None = None

    # The tools this agent's model may ask for (from the tools/ folder).
    # Empty = no tools; the model is called plainly.
    tools: tuple[BaseTool, ...] = ()

    # May this agent hand work to other agents with "HANDOFF: <name>"?
    # If True, `run_agent` tells the model which other agents exist (from
    # the registry, so new agents appear automatically) and watches for the
    # HANDOFF line. See core/handoff.py.
    can_hand_off: bool = False

    # Should the chat show this agent's model output word by word as it is
    # generated (step 8, chat.py)? True for agents that write chat replies.
    # False for agents whose model output is data, not conversation - the
    # Quiz Maker's raw JSON is not something to stream at the student.
    stream_replies: bool = True

    def system_prompt(self) -> str:
        """This agent's prompt text (read from disk once, then cached - see core/prompts.py)."""
        return load_prompt(self.prompt_file)


# ---------------------------------------------------------------------------
# Calling the model, including the tool-calling loop
# ---------------------------------------------------------------------------

def call_model(
    spec: AgentSpec,
    messages: Sequence[AnyMessage],
    extra_instructions: str = "",
) -> AIMessage:
    """
    Send the conversation to the model, as the given agent, and return its
    final reply - running any tools the model asks for along the way.

    What actually gets sent to Ollama is:

        [SystemMessage(<the agent's prompt> + <extra_instructions>),
         <message 1>, <message 2>, ...]

    `extra_instructions` is for information that changes from turn to turn
    and so cannot live in the prompt file - e.g. the list of agents to hand
    off to, or the answer key of the quiz the Tutor is grading.

    The system prompt is added here, on the way out, and is NOT stored in the
    conversation history. The agents share one history, but each one sees
    only its own instructions.

    THE TOOL-CALLING LOOP (only for agents that have tools)
    -------------------------------------------------------
        1. Send the conversation to the model, together with the tools'
           schemas (`bind_tools`), so the model knows which tools exist.
        2. The model replies. Either:
             a) with normal text          -> that is the final answer; stop.
             b) with `tool_calls`, e.g.
                [{"name": "get_topic_facts", "args": {"topic": "gravity"},
                  "id": "call_1"}]        -> the model is ASKING us to run a
                                             tool. It has not run anything.
        3. Our code runs each requested tool (`run_tool`).
        4. Each result goes back as a ToolMessage, linked to the request by
           `tool_call_id`. Both the model's request and the results are added
           to the conversation, so the model can see what it asked for and
           what came back.
        5. Go back to 1. The model now writes its answer using the results
           (or asks for another tool).

    The tool requests and results stay INSIDE this function: only the final
    reply goes into the shared conversation history. The other agents do not
    need to see the Quiz Maker's research notes - but the MLflow trace shows
    every round in full.

    Args:
        spec: the agent doing the talking.
        messages: the conversation so far (from the graph state).

    Returns:
        The model's final reply as an AIMessage, tagged with the agent's name.
    """
    # The logger is named after the agent, so its lines show as [tutor],
    # [quiz_maker], ... and you can see which agent is doing the work.
    log = get_logger(spec.name)

    model = build_chat_model(temperature=spec.temperature)

    # Step 1 of the loop: `bind_tools` returns a copy of the model that sends
    # the tools' schemas along with every request. Without this the model
    # would not know the tools exist.
    model_with_tools = model.bind_tools(list(spec.tools)) if spec.tools else model

    system_text = spec.system_prompt()
    if extra_instructions:
        system_text += "\n\n" + extra_instructions
    conversation = [SystemMessage(content=system_text), *messages]

    # At DEBUG level, print exactly what the model is about to receive.
    # `.type` is "system", "human", "ai" or "tool".
    for message in conversation:
        log.debug("  -> %-6s | %s", message.type, message.content)

    # TRACING: wrap the whole turn in an "agent:<name>" span (see
    # infra/tracing.py). The automatic `ChatOllama` spans for each model call,
    # and our `tool:<name>` spans, land INSIDE this one, so the trace shows
    # which agent made each call. With tracing off, this `with` does nothing.
    with mlflow.start_span(name=f"agent:{spec.name}", span_type=SpanType.AGENT) as span:
        # Attributes are searchable labels shown on the span in the UI.
        span.set_attributes({
            "agent": spec.name,
            "model": model.model,
            "prompt_file": spec.prompt_file,
            "tools": [t.name for t in spec.tools],
        })
        # Inputs: the conversation as the agent sees it, as readable text.
        span.set_inputs({
            "messages": [f"{m.type}: {m.content}" for m in conversation],
        })

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            log.info(
                "calling %s (round %d, %d messages, tools: %s)",
                model.model, round_number, len(conversation),
                ", ".join(t.name for t in spec.tools) or "none",
            )
            started = time.perf_counter()
            reply = model_with_tools.invoke(conversation)          # step 1-2
            log.info("reply received in %.1fs", time.perf_counter() - started)
            reply = _recover_text_tool_call(spec, reply, round_number)

            if not reply.tool_calls:
                break                                              # step 2a: done

            # Step 2b: the model asked for one or more tools.
            conversation.append(reply)
            for call in reply.tool_calls:
                log.info("model requested tool %s(%s)", call["name"], call["args"])
                result = run_tool(spec, call)                      # step 3
                conversation.append(                               # step 4
                    ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])
                )
                log.debug("  <- tool   | %s", result)
            # ...and loop round (step 5).

        else:
            # Python's `for ... else`: this runs only if the loop was NOT
            # ended by `break`, i.e. the model was still asking for tools
            # after MAX_TOOL_ROUNDS. Ask once more WITHOUT tools, which forces
            # a plain-text answer from what it has gathered so far.
            log.warning("still asking for tools after %d rounds; forcing an answer", MAX_TOOL_ROUNDS)
            reply = model.invoke(conversation)

        span.set_outputs({"reply": reply.content, "rounds": round_number})

    log.info("final reply: %d characters", len(reply.content))
    log.debug("  <- ai     | %s", reply.content)

    # Stamp the reply with the agent's name. The model does not care, but it
    # means the saved history (and the MLflow traces) record which agent
    # produced each message.
    reply.name = spec.name
    return reply


def _recover_text_tool_call(spec: AgentSpec, reply: AIMessage, round_number: int) -> AIMessage:
    """
    Fix a known small-model quirk: writing a tool call as plain TEXT.

    When tools are bound, llama3.1 sometimes puts the call in the message text
    instead of the proper `tool_calls` field:

        content = '{"name": "get_topic_facts", "parameters": {"topic": "gravity"}}'
        tool_calls = []

    To our loop that looks like a final answer, so the tool would never run.
    It even does this for things that are NOT tools, like our CLARIFY marker:

        content = '{"name": "CLARIFY", "parameters": {"question": "Which topic?"}}'

    This function spots that shape and converts it back:
        a real tool's name -> a proper tool call, so the loop runs the tool
        "CLARIFY"          -> the text "CLARIFY: <question>", so step 5's
                              clarifying flow recognises it
    Anything else is returned unchanged.
    """
    if reply.tool_calls or not spec.tools:
        return reply

    text = (reply.content or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return reply
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return reply
    if not isinstance(data, dict) or "name" not in data:
        return reply  # e.g. a real quiz JSON, which has "topic", not "name"

    name = str(data["name"])
    args = data.get("parameters", data.get("arguments", {}))
    if not isinstance(args, dict):
        return reply

    log = get_logger(spec.name)
    if name.strip().upper() == "CLARIFY":
        question = next((value for value in args.values() if isinstance(value, str)), "")
        log.warning("model wrote CLARIFY as a fake tool call; converting it")
        return AIMessage(content=f"CLARIFY: {question}")

    if name in {t.name for t in spec.tools}:
        log.warning("model wrote the %s call as text; converting it to a real tool call", name)
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"text-call-{round_number}"}],
        )

    return reply


def run_tool(spec: AgentSpec, call: dict) -> str:
    """
    Run one tool call requested by the model and return the result as text.

    Args:
        spec: the agent whose model made the request (only ITS tools may run).
        call: one entry of `reply.tool_calls`:
              {"name": "get_topic_facts", "args": {"topic": "gravity"}, "id": "..."}

    Errors are RETURNED as text, not raised. Small models sometimes invent a
    tool name or pass bad arguments; telling the model what went wrong lets
    it correct itself on the next round instead of crashing the chat.
    """
    log = get_logger(spec.name)
    tools_by_name = {t.name: t for t in spec.tools}
    tool = tools_by_name.get(call["name"])

    if tool is None:
        log.warning("model asked for unknown tool %r", call["name"])
        return (
            f"Error: there is no tool called {call['name']!r}. "
            f"Available tools: {', '.join(tools_by_name) or 'none'}."
        )

    # TRACING: one span per tool call, labelled with which agent called it.
    with mlflow.start_span(name=f"tool:{tool.name}", span_type=SpanType.TOOL) as span:
        span.set_attributes({"tool": tool.name, "called_by": spec.name})
        span.set_inputs(call["args"])
        try:
            # `.invoke(args)` validates the arguments against the tool's
            # schema (tools/topic_facts.py: TopicFactsInput), then calls the
            # Python function.
            result = str(tool.invoke(call["args"]))
        except Exception as error:  # noqa: BLE001 - report anything back to the model
            log.warning("tool %s failed: %s", tool.name, error)
            result = f"Error while running {tool.name}: {error}"
        span.set_outputs({"result": result})

    return result


# ---------------------------------------------------------------------------
# Turning a reply into state changes
# ---------------------------------------------------------------------------

def clarification_update(spec: AgentSpec, reply: AIMessage) -> dict | None:
    """
    If the reply is "CLARIFY: <question>", return the state changes that
    pause the graph to ask it. Otherwise return None.

    Setting `pending_question` makes the graph route to the ask_user node,
    which pauses (the whole flow is explained in core/clarify.py).
    """
    question = parse_clarification(reply.content)
    if question is None:
        return None

    get_logger(spec.name).info("needs more information, asking: %r", question)

    # Store only the clean question (without the "CLARIFY:" marker) in the
    # history, so the conversation reads naturally on the screen and to the
    # model later on.
    return {
        "messages": [AIMessage(content=question, name=spec.name)],
        "pending_question": {"agent": spec.name, "question": question},
    }


def _handoff_menu(spec: AgentSpec) -> tuple[list[str], str]:
    """
    The other agents this agent may hand off to, and the instructions telling
    the model how. Built from the registry, so a new agent shows up here
    automatically.

    Returns:
        (list of agent names, instruction text to add to the system prompt)
    """
    # Imported here, inside the function, not at the top of the file:
    # agents/registry.py imports the agent files, which import THIS file, so
    # a top-level import would go round in a circle.
    from agents.registry import AGENTS

    others = {name: other for name, other in AGENTS.items() if name != spec.name}
    lines = [f"- {name}: {other.description}" for name, other in others.items()]
    text = (
        "# Other agents you can hand work to\n\n"
        + "\n".join(lines)
        + "\n\nTo hand the student's request to one of them, reply with ONE line "
          "in exactly this format, and nothing else:\n\n"
          "HANDOFF: <agent name> | <the task, with every detail the student gave>"
    )
    return list(others), text


def run_agent(spec: AgentSpec, state: dict, extra_instructions: str = "") -> dict:
    """
    One complete turn for a simple agent: the body of its node function.

    Calls the model, then:
      * "CLARIFY: <question>"  -> pause and ask (clarification_update, step 5)
      * "HANDOFF: <agent>"     -> hand the request to that agent, and get
                                  control back when it finishes (step 7;
                                  only for agents with can_hand_off=True)
      * anything else          -> add the reply to the conversation; done

    Agents that need to do more with the reply (like the Quiz Maker, which
    turns it into a structured quiz) call `call_model` and
    `clarification_update` themselves instead.

    Args:
        extra_instructions: added to the system prompt for this turn only.

    Returns:
        The state changes for LangGraph to apply. A handoff is requested with
        the extra key "handoff", which the wrapper in core/handoff.py turns
        into the real bookkeeping.
    """
    handoff_targets: list[str] = []
    if spec.can_hand_off:
        handoff_targets, menu = _handoff_menu(spec)
        extra_instructions = "\n\n".join(part for part in (menu, extra_instructions) if part)

    reply = call_model(spec, state["messages"], extra_instructions)

    pause = clarification_update(spec, reply)
    if pause:
        return pause

    if handoff_targets:
        target, task, rest = parse_handoff(reply.content, handoff_targets)
        if target:
            # Keep a short visible note in the chat, so the student (and the
            # history) shows what happened.
            note = rest or f"Let me hand this over to {target}."
            return {
                "messages": [AIMessage(content=note, name=spec.name)],
                "pending_question": None,
                # "call": control comes back to this agent afterwards.
                # "task": the job description the receiving agent will read.
                "handoff": {"to": target, "mode": "call", "task": task},
            }
        reply.content = rest  # drop an ignored HANDOFF line, if any

    return {"messages": [reply], "pending_question": None}
