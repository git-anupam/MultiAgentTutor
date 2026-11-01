"""
infra/postgres.py - saves every conversation to PostgreSQL.

BACKGROUND: WHAT IS A CHECKPOINTER?
-----------------------------------
In step 2 the graph forgot everything the moment a run finished, so the demo
script had to keep the history itself and pass it back in every time.

A CHECKPOINTER fixes that. After every step of the graph, LangGraph hands the
whole state (see core/state.py) to the checkpointer, which writes a snapshot
- a "checkpoint" - to the database. On the next run, LangGraph asks the
checkpointer for the latest snapshot and continues from there.

    run 1:  load (nothing) -> tutor -> save
    run 2:  load (run 1's state) -> tutor -> save
    ...quit the program, reboot, come back tomorrow...
    run 3:  load (run 2's state) -> tutor -> save

THREADS: ONE SAVED CONVERSATION EACH
------------------------------------
Every checkpoint is filed under a THREAD ID, a name you choose. Each thread
is a separate conversation:

    graph.invoke(input, config={"configurable": {"thread_id": "biology"}})

Use the same thread id again later and you resume that conversation; use a new
one and you start fresh.

WHY THIS MATTERS FOR CLARIFYING QUESTIONS
-----------------------------------------
An agent can PAUSE in the middle of a run to ask you a clarifying question
(core/clarify.py). The paused run is also saved as a checkpoint, so you can quit while
it is waiting for your answer, restart, and answer then.

WHAT LANGGRAPH GIVES US
-----------------------
`PostgresSaver` (from the `langgraph-checkpoint-postgres` package) does all
the real work: creating tables, serialising messages, loading the latest
snapshot. This file only adds a clear connection check and a friendly error
message, so a stopped database does not show up as a 40-line traceback.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from core.config import get_settings
from core.logging_setup import get_logger

log = get_logger("postgres")


class PostgresNotReachable(RuntimeError):
    """Raised when the database does not accept a connection."""


def _location(uri: str) -> str:
    """
    "postgresql://user:pass@localhost:5433/studyhelper" -> "localhost:5433/studyhelper"

    Used in log lines, so the password never gets printed.
    """
    return uri.rsplit("@", 1)[-1]


def check_connection(timeout: int = 5) -> None:
    """
    Make sure Postgres is up before we hand it to LangGraph.

    Opens a connection, runs the simplest possible query, and closes it again.

    Raises:
        PostgresNotReachable: with instructions for starting the database.
    """
    uri = get_settings().postgres_uri
    try:
        # `connect_timeout` stops us from hanging forever when the database
        # is not running. `with` closes the connection when the block ends.
        with psycopg.connect(uri, connect_timeout=timeout) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError as error:
        raise PostgresNotReachable(
            f"Could not connect to PostgreSQL at {_location(uri)}.\n"
            f"Underlying error: {error}\n\n"
            "Things to try (from the project folder):\n"
            "  1. Start it:          docker compose up -d\n"
            "  2. Check it is up:    docker compose ps    (look for 'healthy')\n"
            "  3. Read its log:      docker compose logs db\n"
        ) from error

    log.info("connected to %s", _location(uri))


@contextmanager
def open_checkpointer() -> Iterator[PostgresSaver]:
    """
    Open a Postgres checkpointer for the duration of a `with` block.

    Usage:
        with open_checkpointer() as checkpointer:
            graph = build_graph(checkpointer)
            ...chat...
        # the database connection is closed here, even if an error happened

    WHY A `with` BLOCK?
    A checkpointer holds an open database connection. A `with` block
    guarantees that connection is closed when we are done - even if the
    program crashes or you press Ctrl+C - instead of being left dangling.
    """
    check_connection()

    uri = get_settings().postgres_uri
    with PostgresSaver.from_conn_string(uri) as checkpointer:
        # `.setup()` creates LangGraph's tables the first time, and does
        # nothing on later runs, so it is safe to call every time. The tables
        # it creates:
        #   checkpoints          one row per saved snapshot
        #   checkpoint_blobs     the larger values inside a snapshot (messages)
        #   checkpoint_writes    in-progress writes from a step
        #   checkpoint_migrations  which table-layout version is installed
        checkpointer.setup()
        log.info("checkpoint tables ready")
        yield checkpointer


# ---------------------------------------------------------------------------
# Thread helpers
#
# Every time the chat starts, it begins a NEW conversation with a new thread
# id, unless you ask to resume an old one with `--thread <id>`. These two
# functions make that possible: one invents a fresh id, the other lists the
# ids already saved so you can pick one to resume.
# ---------------------------------------------------------------------------

def new_thread_id() -> str:
    """
    Invent an id for a brand-new conversation, e.g. "chat-20261101-143210".

    Based on the current date and time, so ids are readable, sort in the
    order the conversations were started, and never repeat (unless you start
    two chats within the same second).
    """
    return f"chat-{datetime.now():%Y%m%d-%H%M%S}"


def list_threads() -> list[dict]:
    """
    List every saved conversation, most recently active first.

    Call this inside `with open_checkpointer():`, so the tables are
    guaranteed to exist.

    Returns:
        One dict per thread: {"thread_id", "checkpoints", "last_active"}.

    HOW IT WORKS
    This reads LangGraph's `checkpoints` table directly with SQL (the same
    query you can run yourself in psql or the VS Code database viewer):
      * `checkpoint->>'ts'` is the time a snapshot was saved, stored inside
        the snapshot's JSON column
      * `checkpoint_ns = ''` keeps only the main graph's snapshots
    """
    query = """
        SELECT thread_id,
               count(*)               AS checkpoints,
               max(checkpoint->>'ts') AS last_active
        FROM checkpoints
        WHERE checkpoint_ns = ''
        GROUP BY thread_id
        ORDER BY last_active DESC
    """
    with psycopg.connect(get_settings().postgres_uri) as conn:
        rows = conn.execute(query).fetchall()

    return [
        {"thread_id": thread_id, "checkpoints": count, "last_active": last_active}
        for thread_id, count, last_active in rows
    ]
