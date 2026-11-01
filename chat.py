"""
chat.py - the terminal chat interface for the study helper (step 8).

Run it from the project folder (with Postgres running: `docker compose up -d`):

    python chat.py                       start a NEW conversation
    python chat.py --thread <id>         resume a saved conversation
    python chat.py --list                list saved conversations
    python chat.py --quiet               hide the flow log, show only the chat

Inside the chat, type /help for the commands.

WHAT THIS FILE ADDS: STREAMING
------------------------------
`scripts/try_checkpoint.py` calls `graph.invoke(...)`, which waits for the
whole run to finish and then returns the final state. With a local model
that can mean a minute of silence.

This file calls `graph.stream(...)` instead, which hands us events WHILE the
graph runs. We ask for two kinds of event at once:

    "messages"  each small piece (token) of text as the model generates it,
                with metadata saying which node is running the model
    "updates"   each node's state changes, as soon as that node finishes

so the Tutor's answer appears word by word, and messages that do not come
from a model (the Tutor presenting a quiz, the Quiz Maker's "I made a
quiz") are printed the moment their node finishes.

NOT EVERYTHING SHOULD BE STREAMED
---------------------------------
Several model calls produce text the student should never see:

    router           its JSON decision         -> skipped: not an agent node
    quiz_maker       its quiz JSON             -> skipped: stream_replies=False
    tutor            "HANDOFF: quiz_maker"     -> held back (see ReplyStreamer)
    any agent        "CLARIFY: ..."            -> held back; shown as a question

The first two are decided by WHICH node is running. The last two can only be
spotted from the reply's first few characters, so `ReplyStreamer` holds
those back until it can tell.
"""

from __future__ import annotations

# --- Silence one harmless warning, BEFORE LangGraph is imported -------------
# LangGraph's checkpoint module triggers a "pending deprecation" warning from
# LangChain about a future default ("allowed_objects"). It does not affect
# us, and it appears before the chat even starts, which is confusing.
#
# The ORDER matters, twice over:
#   1. Importing `langchain_core` installs its own filter that switches these
#      warnings back ON. Python checks the newest filter first, so ours must
#      be added AFTER that import, or LangChain's would win.
#   2. A filter only affects warnings raised after it is set, so ours must
#      come BEFORE the imports below, which trigger the warning.
import warnings

import langchain_core  # noqa: E402, F401  (imported for its side effect; see above)

warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

from agents.registry import AGENTS  # noqa: E402
from core.clarify import graph_input, pending_question  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.graph import build_graph  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from infra.postgres import (  # noqa: E402
    PostgresNotReachable,
    list_threads,
    new_thread_id,
    open_checkpointer,
)
from infra.tracing import chat_turn_span, setup_tracing  # noqa: E402

log = get_logger("chat")

HELP = """
Commands:
  /help        show this help
  /new         start a new conversation (the current one stays saved)
  /threads     list saved conversations
  /history     show this conversation so far
  /quiz        peek at the latest quiz WITH its answers
  /quit        leave (also: quit, exit, Ctrl+C, Ctrl+D)
"""


# ---------------------------------------------------------------------------
# 1. Colours
#
# ANSI escape codes like "\033[32m" switch the terminal's text colour. We
# only use them when printing to a real terminal (`isatty()`), so redirecting
# the output to a file gives clean text.
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()
_AGENT_COLORS = ["32", "35", "36", "33", "34"]  # green, magenta, cyan, yellow, blue


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def dim(text: str) -> str:
    return _paint(text, "2")


def agent_label(name: str) -> str:
    """The agent's name in bold, in its own colour (by position in the registry)."""
    index = list(AGENTS).index(name) if name in AGENTS else 0
    return _paint(name, "1;" + _AGENT_COLORS[index % len(_AGENT_COLORS)])


# ---------------------------------------------------------------------------
# 1b. Keeping log lines and streamed text apart
#
# Streamed tokens are printed WITHOUT a newline (that is what makes the text
# grow along the line). If a log line is printed at that moment, it gets
# glued onto the end of the reply:   "...today?12:56:23 [tutor] reply..."
#
# `_mid_line` remembers whether the cursor is in the middle of a streamed
# line. The logging filter below runs before every log line; if we are
# mid-line, it ends the line first.
# ---------------------------------------------------------------------------

_mid_line = False


def _set_mid_line(value: bool) -> None:
    global _mid_line
    _mid_line = value


class _NewlineBeforeLogs(logging.Filter):
    """A logging filter that never filters anything out - it only adds a newline."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _mid_line:
            print(flush=True)
            _set_mid_line(False)
        return True


def keep_logs_off_streamed_lines() -> None:
    """Attach the filter to the project's log handler (call after setup_logging)."""
    for handler in logging.getLogger("studyhelper").handlers:
        handler.addFilter(_NewlineBeforeLogs())


# ---------------------------------------------------------------------------
# 2. Printing a reply as it streams in
# ---------------------------------------------------------------------------

class ReplyStreamer:
    """
    Prints one agent's reply token by token - after a short look-ahead.

    The problem: if the Tutor's reply is "HANDOFF: quiz_maker" or
    "CLARIFY: Which topic?", we must not print it raw. But tokens arrive a few
    characters at a time, so when "HAN" arrives we cannot know yet.

    The solution: a small state machine for each reply.

        holding  collect text until there are 7 visible characters
                 (the length of "HANDOFF" and of "CLARIFY")
           |
           +--> starts with HANDOFF / CLARIFY / "{"  ->  hidden
           |       (the node's final message is printed properly later)
           |
           +--> anything else  ->  live: print what we have, then print
                                   every further token immediately
    """

    MARKERS = ("HANDOFF", "CLARIFY", "{")   # "{" = JSON, e.g. a text tool call
    LOOK_AHEAD = 7                          # len("HANDOFF") == len("CLARIFY")

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.node: str | None = None
        self.buffer = ""
        self.state = "idle"     # idle | holding | live | hidden

    def _head(self) -> str:
        """The buffer without leading spaces or Markdown, upper-cased."""
        return self.buffer.lstrip(" \t\n*_#>-").upper()

    def feed(self, node: str, text: str) -> None:
        """Handle one streamed token from `node`."""
        if node != self.node:          # a new reply begins
            self.finish()
            self.node, self.state = node, "holding"

        if self.state == "hidden":
            return
        if self.state == "live":
            print(text, end="", flush=True)
            _set_mid_line(True)
            return

        # state == "holding"
        self.buffer += text
        head = self._head()
        if head.startswith(self.MARKERS):
            self.state = "hidden"
        elif len(head) >= self.LOOK_AHEAD:
            self.state = "live"
            print(f"\n{agent_label(node)}: {self.buffer.lstrip()}", end="", flush=True)
            _set_mid_line(True)

    def finish(self) -> bool:
        """
        End the current reply.

        Returns:
            True if its text reached the screen (so the caller must not print
            the same message again).
        """
        shown = False
        if self.state == "live":
            if _mid_line:           # a log line may already have ended it
                print()
                _set_mid_line(False)
            shown = True
        elif self.state == "holding" and self.buffer.strip():
            # A reply shorter than the look-ahead ("Hi!") that is not a marker.
            print(f"\n{agent_label(self.node)}: {self.buffer.strip()}")
            shown = True
        self._reset()
        return shown

    def finish_node(self, node: str) -> bool:
        """`finish()`, but only if `node`'s reply is the one in progress."""
        return self.finish() if self.node == node else False


def show_node_update(node: str, update: dict, streamer: ReplyStreamer) -> None:
    """
    Print an agent's new messages when its node finishes - unless they were
    already streamed, or are a clarifying question (shown after the run).
    """
    already_shown = streamer.finish_node(node)
    if update.get("pending_question"):
        return
    for message in update.get("messages") or []:
        if message.type == "ai" and not already_shown:
            print(f"\n{agent_label(message.name or node)}: {message.content}")


# ---------------------------------------------------------------------------
# 3. One turn: the user's message in, the agents' replies out
# ---------------------------------------------------------------------------

def run_turn(graph, config: dict, user_text: str, quiet: bool) -> None:
    """Run the graph for one user message, streaming the replies."""
    thread_id = config["configurable"]["thread_id"]

    # As in try_checkpoint.py: is this the answer to a clarifying question?
    answering = pending_question(graph, config) is not None
    count_before = len(graph.get_state(config).values.get("messages", []))

    if quiet:
        # With the log hidden, give some sign of life during long waits.
        print(dim("  ...thinking"), flush=True)

    streamer = ReplyStreamer()

    # One MLflow trace per user message (infra/tracing.py).
    with chat_turn_span(thread_id, user_text, answers_question=answering) as span:

        # stream_mode=[...] with two modes yields (mode, data) pairs.
        for mode, data in graph.stream(
            graph_input(graph, config, user_text),
            config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                # data = (message chunk, metadata). The metadata says which
                # graph node the model call belongs to.
                chunk, metadata = data
                spec = AGENTS.get(metadata.get("langgraph_node"))
                if (
                    spec is not None
                    and spec.stream_replies
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    streamer.feed(spec.name, chunk.content)

            else:  # mode == "updates"
                # data = {node name: the state changes that node returned}.
                # Also contains "router", "ask_user" and "__interrupt__"
                # entries, which have nothing to print.
                for node, update in data.items():
                    if node in AGENTS and isinstance(update, dict):
                        show_node_update(node, update, streamer)

        streamer.finish()

        # Record the outcome on the trace, as try_checkpoint.py does.
        messages = graph.get_state(config).values.get("messages", [])
        new_replies = [m for m in messages[count_before:] if m.type == "ai"]
        question = pending_question(graph, config)
        span.set_outputs({
            # This message's replies, and which agents produced them.
            "replies": [f"{m.name}: {m.content}" for m in new_replies],
            "agents_involved": [m.name for m in new_replies],
            "paused_for_clarification": question is not None,
            # The WHOLE conversation so far, each line labelled with who
            # wrote it. One trace covers one message, but this lets the
            # latest trace show the full story of the session:
            # tutor -> quiz_maker -> tutor ... (The Sessions page in the
            # MLflow UI lists every trace of the conversation in order.)
            "conversation": [
                f"{'You' if m.type == 'human' else (m.name or m.type)}: {m.content}"
                for m in messages
            ],
        })

    log.info("saved: thread now has %d messages", len(messages))

    if question:
        print(f"\n{agent_label(question['agent'])} asks: {question['question']}")
        print(dim("  (paused - type your answer, or /quit and resume later)"))
    print()


# ---------------------------------------------------------------------------
# 4. Commands and small printing helpers
# ---------------------------------------------------------------------------

def print_history(messages) -> None:
    for message in messages:
        who = "You" if message.type == "human" else agent_label(message.name or message.type)
        text = message.content.replace("\n", " ")
        print(f"  {who}: {text[:150]}{'...' if len(text) > 150 else ''}")


def print_quiz_with_answers(quiz: dict | None) -> None:
    if not quiz:
        print("  (no quiz yet)\n")
        return
    print(f"\n  Quiz: {quiz['topic']}")
    for number, item in enumerate(quiz["questions"], start=1):
        print(f"\n  {number}. {item['question']}")
        for index, choice in enumerate(item["choices"]):
            print(f"     {chr(ord('A') + index)}) {choice}")
        print(dim(f"     answer: {item['answer']} - {item['explanation']}"))
    print()


def print_threads() -> None:
    threads = list_threads()
    if not threads:
        print("\n  No saved conversations yet.\n")
        return
    print(f"\n  {'thread id':<24}  {'last active (UTC)':<19}  checkpoints")
    for thread in threads:
        last_active = (thread["last_active"] or "")[:19].replace("T", " ")
        print(f"  {thread['thread_id']:<24}  {last_active:<19}  {thread['checkpoints']}")
    print("\n  Resume one with:  python chat.py --thread <thread id>\n")


def show_start(graph, config: dict, resumed_requested: bool) -> None:
    """Print the header, plus the old conversation when resuming."""
    thread_id = config["configurable"]["thread_id"]
    existing = graph.get_state(config).values.get("messages", [])

    if existing:
        print(f"\nResuming conversation {thread_id}:")
        print_history(existing)
    elif resumed_requested:
        print(f"\nNo saved conversation called '{thread_id}' - starting it fresh.")

    print(dim(
        f"\nModel: {get_settings().ollama_model} | thread: {thread_id} | "
        f"agents: {', '.join(AGENTS)} | /help for commands\n"
    ))

    waiting = pending_question(graph, config)
    if waiting:
        print(f"{agent_label(waiting['agent'])} is waiting for your answer: {waiting['question']}\n")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent study helper.")
    parser.add_argument("--thread", help="Resume the conversation with this thread id.")
    parser.add_argument("--list", action="store_true", help="List saved conversations and exit.")
    parser.add_argument("--quiet", action="store_true", help="Hide the flow log.")
    args = parser.parse_args()

    setup_logging("WARNING" if args.quiet else None)
    keep_logs_off_streamed_lines()
    setup_tracing()

    config = {"configurable": {"thread_id": args.thread or new_thread_id()}}

    try:
        with open_checkpointer() as checkpointer:
            if args.list:
                print_threads()
                return 0

            graph = build_graph(checkpointer)
            show_start(graph, config, resumed_requested=bool(args.thread))

            while True:
                try:
                    user_text = input(_paint("You: ", "1")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_text:
                    continue
                command = user_text.lower()
                if command in {"/quit", "quit", "exit"}:
                    break
                if command == "/help":
                    print(HELP)
                elif command == "/new":
                    config = {"configurable": {"thread_id": new_thread_id()}}
                    show_start(graph, config, resumed_requested=False)
                elif command == "/threads":
                    print_threads()
                elif command == "/history":
                    print_history(graph.get_state(config).values.get("messages", []))
                    print()
                elif command == "/quiz":
                    print_quiz_with_answers(graph.get_state(config).values.get("quiz"))
                elif command.startswith("/"):
                    print(f"  Unknown command {user_text!r}. Type /help.\n")
                else:
                    try:
                        run_turn(graph, config, user_text, quiet=args.quiet)
                    except KeyboardInterrupt:
                        # Ctrl+C while the model is working: abandon this
                        # reply but keep the chat open. Postgres still has
                        # the last checkpoint saved before the interruption.
                        print(dim("\n  (interrupted - reply abandoned)\n"))

    except PostgresNotReachable as error:
        print(f"\n{error}")
        return 1

    thread_id = config["configurable"]["thread_id"]
    print(f"Bye! To resume:  python chat.py --thread {thread_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
