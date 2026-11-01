"""
scripts/try_checkpoint.py - the full multi-agent chat WITHOUT streaming.

The real interface is `chat.py`. This script does the same job with
`graph.invoke(...)`, which waits for the whole run and returns the final
state, instead of `graph.stream(...)`. It is kept as a simpler version to
compare with, and for its extra `/checkpoints` command, which lists every
snapshot Postgres holds for the conversation.

Start Postgres first (once):

    docker compose up -d

Then:

    python scripts/try_checkpoint.py                             # NEW conversation
    python scripts/try_checkpoint.py --list                      # list saved ones
    python scripts/try_checkpoint.py --thread chat-20261101-143210   # resume one

Commands inside the chat:

    /history      show the saved conversation
    /checkpoints  list every snapshot saved for this thread
    /quiz         show the latest quiz WITH its answers (a peek for learning)
    quit          leave (Ctrl+C and Ctrl+D also work)

WHAT TO TRY
-----------
1. Start the script. It prints the new conversation's thread id. Tell the
   Tutor your name, then `quit`.
2. Start it again with no arguments: a NEW conversation. The Tutor does not
   know your name - different thread, different memory.
3. Run with `--list`, copy the first conversation's id, and run with
   `--thread <that id>`. The old conversation is printed and the Tutor
   remembers your name - the history came back from Postgres.
4. (Step 5) In a NEW conversation, type just "explain it". The Tutor asks
   which topic, and the graph PAUSES. Answer it, and the Tutor explains.
5. (Step 5) Do it again, but `quit` while the question is waiting. Resume
   with `--thread <id>`: the script shows the waiting question, and your
   answer continues the paused run.
6. (Step 7) Type "make me a quiz on gravity". Watch the log: the router
   picks the Tutor, the Tutor hands off to the Quiz Maker (a "call"), the
   Quiz Maker uses its tool and writes the quiz, control returns to the
   Tutor, and the Tutor shows you the questions.
7. (Step 7) Answer, e.g. "1-B, 2-A, 3-C". The router picks the Tutor, which
   grades you using the hidden answer key.
8. (Step 7) Type "make it harder". The router switches straight to the Quiz
   Maker, which writes a new quiz and then TRANSFERS to the Tutor to show it.
9. (Step 7) In a new conversation, type just "make me a quiz": the Quiz
   Maker - working for the Tutor - pauses to ask for a topic.

ONLY THE NEW MESSAGE GOES IN
----------------------------
Each `graph.invoke(...)` gets only what you just typed. The graph loads the
earlier messages from the checkpoint itself, and the `add_messages` reducer
(core/state.py) appends the new one. And when the thread is paused on a
question, what you type is sent as the ANSWER instead - see
`graph_input()` in core/clarify.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import core...` work when run as `python scripts/try_checkpoint.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

log = get_logger("checkpoint")


def print_history(messages) -> None:
    """Print a saved conversation, one line per message."""
    for message in messages:
        # Human messages have no name; agent replies carry the agent's name
        # (stamped on them in agents/base.py).
        who = "You" if message.type == "human" else (message.name or message.type)
        text = message.content.replace("\n", " ")
        print(f"  {who}: {text[:150]}{'...' if len(text) > 150 else ''}")


def print_quiz(quiz: dict | None, show_answers: bool) -> None:
    """
    Print a quiz from the state. Questions only by default; `/quiz` also
    shows the answers, so you can see what the Quiz Maker stored.
    """
    if not quiz:
        print("  (no quiz yet)")
        return
    print(f"\n  Quiz: {quiz['topic']}")
    for number, item in enumerate(quiz["questions"], start=1):
        print(f"\n  {number}. {item['question']}")
        for index, choice in enumerate(item["choices"]):
            print(f"     {chr(ord('A') + index)}) {choice}")
        if show_answers:
            print(f"     answer: {item['answer']} - {item['explanation']}")
    print()


def print_threads() -> None:
    """Print every saved conversation, most recent first (the --list option)."""
    threads = list_threads()
    if not threads:
        print("\nNo saved conversations yet.")
        return

    print("\nSaved conversations (most recent first, times in UTC):\n")
    print(f"  {'thread id':<24}  {'last active':<19}  checkpoints")
    for thread in threads:
        # The timestamp looks like "2026-11-01T14:32:10.123456+00:00";
        # keep "2026-11-01 14:32:10".
        last_active = (thread["last_active"] or "")[:19].replace("T", " ")
        print(f"  {thread['thread_id']:<24}  {last_active:<19}  {thread['checkpoints']}")
    print("\nResume one with:  python scripts/try_checkpoint.py --thread <thread id>")


def print_checkpoints(graph, config) -> None:
    """
    List every snapshot saved for this thread, newest first.

    You will notice there are MORE checkpoints than messages. LangGraph saves
    one snapshot when your message enters the graph (source="input") and one
    after every node finishes (source="loop"). That fine-grained saving is
    what lets a run pause in the middle (step 5) and resume exactly there.

    The `next` column says which node would run next from that snapshot; it
    is empty once a run has finished. While a clarifying question is
    waiting, the newest snapshot shows `next = ask_user`: the paused run.
    """
    print("  step  source  messages  next      checkpoint id")
    for snapshot in graph.get_state_history(config):
        meta = snapshot.metadata or {}
        print(
            f"  {meta.get('step', '?'):>4}  {meta.get('source', '?'):<6}  "
            f"{len(snapshot.values.get('messages', [])):>8}  "
            f"{', '.join(snapshot.next) or '-':<8}  "
            f"{snapshot.config['configurable']['checkpoint_id']}"
        )


def chat_loop(graph, config) -> None:
    """Read a line, run the graph once (traced), print the reply, repeat."""
    thread_id = config["configurable"]["thread_id"]

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            return
        if user_text == "/history":
            print_history(graph.get_state(config).values.get("messages", []))
            continue
        if user_text == "/checkpoints":
            print_checkpoints(graph, config)
            continue
        if user_text == "/quiz":
            print_quiz(graph.get_state(config).values.get("quiz"), show_answers=True)
            continue

        # Is the graph paused, waiting for an answer to a clarifying question?
        # Then what the user just typed is that answer (step 5).
        answering = pending_question(graph, config) is not None

        # Count the messages so far, to find the ones this turn adds. With
        # handoffs, one turn can add several replies from different agents.
        count_before = len(graph.get_state(config).values.get("messages", []))

        # Everything inside this `with` becomes ONE trace in MLflow
        # (step 4, see infra/tracing.py).
        with chat_turn_span(thread_id, user_text, answers_question=answering) as span:
            # graph_input() gives either a new message or Command(resume=...)
            # for an answer. The `config` tells LangGraph which thread to load
            # from and save to.
            result = graph.invoke(graph_input(graph, config, user_text), config)

            # The agents' messages from this turn, in order.
            new_replies = [m for m in result["messages"][count_before:] if m.type == "ai"]
            question = pending_question(graph, config)

            # `result["messages"]` is the WHOLE conversation (loaded from the
            # checkpoint + this turn), so record it on the top-level span.
            # That way, opening a trace in MLflow shows the full conversation
            # up to and including this message, not only the latest reply.
            span.set_outputs({
                "replies": [f"{m.name}: {m.content}" for m in new_replies],
                "agents_involved": [m.name for m in new_replies],
                "paused_for_clarification": question is not None,
                "conversation": [
                    f"{'You' if m.type == 'human' else (m.name or m.type)}: {m.content}"
                    for m in result["messages"]
                ],
            })

        # By the time `invoke` returns, the checkpointer has already written
        # the new state to Postgres. `get_state` reads the latest snapshot
        # back, so we can show its id in the log.
        saved = graph.get_state(config)
        log.info(
            "saved: thread now has %d messages (checkpoint %s)",
            len(saved.values["messages"]),
            saved.config["configurable"]["checkpoint_id"],
        )

        # Print every agent message from this turn, so handoffs are visible:
        #   tutor: Let me hand this over to quiz_maker.
        #   quiz_maker: I made a 3-question quiz on gravity.
        #   tutor: Here's your quiz on gravity! ...
        # If the run paused, the last message is the question; show it
        # separately below.
        shown = new_replies[:-1] if question else new_replies
        for message in shown:
            print(f"\n{message.name}: {message.content}")
        print()

        if question:
            print(f"{question['agent']} asks: {question['question']}")
            print("  (the conversation is paused - type your answer, or quit and resume later)\n")


def main() -> int:
    setup_logging()
    setup_tracing()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Tutor chat saved to Postgres.")
    parser.add_argument(
        "--thread",
        help="Resume the saved conversation with this thread id. "
             "Leave out to start a new conversation.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List saved conversations and exit.",
    )
    args = parser.parse_args()

    # No --thread given = a brand-new conversation with a fresh id. Because
    # the MLflow session is set to the thread id (infra/tracing.py), a new
    # conversation is automatically a new MLflow session too.
    thread_id = args.thread or new_thread_id()

    # This dict travels with every graph call. "configurable" is LangGraph's
    # bag of per-run settings; "thread_id" is the one the checkpointer needs.
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with open_checkpointer() as checkpointer:
            if args.list:
                print_threads()
                return 0

            graph = build_graph(checkpointer)

            # Ask the checkpointer for this thread's latest snapshot. For a
            # thread that has never been used, `.values` is an empty dict.
            existing = graph.get_state(config).values.get("messages", [])
            if existing:
                log.info("thread %r: resumed, %d messages loaded", thread_id, len(existing))
                print(f"\nResuming conversation '{thread_id}':")
                print_history(existing)
            elif args.thread:
                # Asked to resume, but nothing is saved under that id - most
                # likely a typo. Say so rather than silently starting fresh.
                log.warning("thread %r: nothing saved under this id", thread_id)
                print(f"\nNo saved conversation called '{thread_id}' - starting it fresh.")
                print("(Use --list to see the saved ones.)")
            else:
                log.info("thread %r: new conversation", thread_id)

            print(
                f"\nModel: {settings.ollama_model} | thread: {thread_id}\n"
                "Commands: /history /checkpoints /quiz quit\n"
            )

            # Resuming a conversation that was paused on a question? Remind
            # the user what they were asked - their next line answers it.
            waiting = pending_question(graph, config)
            if waiting:
                log.info("thread %r is paused on a clarifying question", thread_id)
                print(f"{waiting['agent']} is still waiting for your answer:")
                print(f"  {waiting['question']}\n")

            chat_loop(graph, config)

    except PostgresNotReachable as error:
        print(f"\n{error}")
        return 1

    print(f"Bye! To resume:  python scripts/try_checkpoint.py --thread {thread_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
