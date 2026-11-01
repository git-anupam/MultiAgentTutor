# What checkpointing does

Without a checkpointer, a LangGraph graph forgets everything when a run
finishes. A **checkpointer** saves the graph's **state** (see
`core/state.py`) after every step, so the next run can continue where the
last one stopped.

This project uses LangGraph's PostgreSQL checkpointer (`PostgresSaver`),
set up in `infra/postgres.py`. Postgres runs in Docker
(`docker-compose.yml`, port 5433).

## Threads

Every snapshot is filed under a **thread id**, which is one conversation:

```python
graph.stream(input, {"configurable": {"thread_id": "chat-20261101-134127"}})
```

| Command | Effect |
| --- | --- |
| `python chat.py` | New conversation, new thread id (printed at the top) |
| `python chat.py --thread <id>` | Resume that conversation, with its full memory |
| `python chat.py --list` | List saved conversations |

Different threads never share memory.

## What gets saved, and when

LangGraph saves a snapshot when your message enters the graph, and again
**after every node**. One quiz request therefore creates several:

```
input   your message arrives
loop    after router
loop    after quiz_maker
loop    after tutor          <- the run is finished
```

Run `python scripts/try_checkpoint.py --thread <id>` and type
`/checkpoints` to see them, including which node would run next from each.

Because the chat passes only the **new** message to the graph, it relies on
the checkpoint for everything else: the earlier messages, the stored quiz,
and any pending question.

## Why it matters for pausing

When an agent asks a clarifying question, the run stops halfway, inside the
`ask_user` node. That half-finished run is a checkpoint too, so you can quit,
come back days later, and your answer continues it. See
[clarifying questions](04_clarifying_questions.md).

## Looking at the tables

```bash
docker exec -it studyhelper-db psql -U studyhelper -c \
  "SELECT thread_id, count(*) FROM checkpoints GROUP BY thread_id;"
```

| Table | Contents |
| --- | --- |
| `checkpoints` | One row per snapshot, linked to its parent snapshot |
| `checkpoint_blobs` | The state values themselves (binary, not readable as text) |
| `checkpoint_writes` | In-progress writes, e.g. a paused run |
| `checkpoint_migrations` | The table-layout version |

To wipe all saved conversations: `docker compose down -v && docker compose up -d`.
