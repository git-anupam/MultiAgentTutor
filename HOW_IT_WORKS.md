# How it works: one message, start to finish

This walks through **one real message** from a test run, step by step: from
the moment you press Enter to the moment the quiz appears. Log lines are
copied from that run (model `qwen2.5:7b`, CPU only).

The conversation so far: you asked *"what is photosynthesis?"* and the Tutor
answered. That is 2 messages, saved in Postgres under thread
`chat-20261101-135805`. Now you type:

```
You: help me build a quiz on gravity
```

---

## The whole journey at a glance

```
 chat.py                LangGraph graph (core/graph.py)                       Postgres   MLflow
 ───────                ───────────────────────────────                       ────────   ──────
 read input ──┐
              ├─ open trace "chat_turn" ─────────────────────────────────────────────────── ●
              └─ graph.stream(new message, thread_id)
                     │
                     ├─ load thread's latest checkpoint (2 messages) ◄─────────── ●
                     ├─ append your message, save ──────────────────────────────► ●
                     │
                     ├─ router ── model picks "quiz_maker" ────────────────────► ●   span
                     │
                     ├─ quiz_maker
                     │     ├─ find_topic ── "gravity"                                 span
                     │     ├─ model asks for tool get_topic_facts                     span
                     │     ├─ our code runs the tool                                  span
                     │     ├─ model writes the quiz JSON                              span
                     │     ├─ validate, store in state["quiz"]
                     │     └─ handoff: transfer to tutor ──────────────────────► ●   span
                     │
                     └─ tutor ── presents the quiz (plain Python) ─────────────► ●
                     │
 print replies ◄─────┘
 close trace ────────────────────────────────────────────────────────────────────────────── ●
```

Now each step in detail.

---

## Step 1: Input (`chat.py`)

The main loop in `chat.py` reads your line with `input()`. It is not a
command (`/help`, `/quiz`, ...), so it calls `run_turn()`.

`run_turn()` first asks: **is this conversation paused on a clarifying
question?** (`pending_question()` in `core/clarify.py`). It is not, so your
text becomes a normal new message (`graph_input()`):

```python
{"messages": [HumanMessage("help me build a quiz on gravity")]}
```

Only the **new** message is sent. The graph gets everything else from the
checkpoint.

## Step 2: The trace starts (`infra/tracing.py`)

`run_turn()` opens a `chat_turn` span with `chat_turn_span(...)`. Everything
that happens until the reply is ready becomes a child of this span, so this
whole message ends up as **one trace** in MLflow, tagged with the thread id.

## Step 3: The graph starts, and the checkpoint is loaded (`infra/postgres.py`)

```python
graph.stream(input, {"configurable": {"thread_id": "chat-20261101-135805"}},
             stream_mode=["messages", "updates"])
```

LangGraph asks the **Postgres checkpointer** for this thread's latest
snapshot: the 2 earlier messages, plus the other state fields
(`core/state.py`). The `add_messages` reducer **appends** your new message,
giving 3 messages, and LangGraph saves an `input` checkpoint.

## Step 4: The router picks an agent (`core/router.py`)

Every new message starts at the `router` node (`START -> router` in
`core/graph.py`). It builds a prompt from `agents/prompts/router.md`, the
**descriptions of all agents in the registry**, the earlier messages, and
your latest message, set apart. Ollama's structured output forces the
answer into `{"reason": ..., "agent": ...}`:

```
14:00:29 [router]  -> quiz_maker (The student requested help in building a quiz, which is a task for the quiz_maker.)
```

The router returns `Command(goto="quiz_maker", ...)`. That both moves the
graph to the Quiz Maker and resets the per-message handoff bookkeeping
(`return_stack = []`, `next_agent = None`, `task = None`). Checkpoint saved.

## Step 5: The Quiz Maker finds the topic (`agents/quiz_maker.py`)

Every agent node is wrapped by `agent_node()` (`core/handoff.py`), which
handles handoff bookkeeping before and after the agent's own code. There is
nothing to do on arrival here.

`quiz_maker_node()` first runs `find_topic()`: a short model call whose reply
can only be `{"topic": "..."}`:

```
14:00:51 [quiz_maker] topic: 'gravity'
```

A topic was found, so **no clarifying question**. (With an empty topic, the
code would ask instead; see the example at the end.)

## Step 6: The tool-calling loop (`agents/base.py`, `tools/topic_facts.py`)

`call_model()` sends the conversation to the model with the Quiz Maker's
prompt (`agents/prompts/quiz_maker.md`), a "Your current task: Topic:
gravity" note, and the **tool schema** attached via `bind_tools`.

**Round 1.** The model does not answer. It *requests* a tool:

```
14:00:51 [quiz_maker] calling qwen2.5:7b (round 1, 4 messages, tools: get_topic_facts)
14:01:49 [quiz_maker] reply received in 58.9s
14:01:49 [quiz_maker] model requested tool get_topic_facts({'topic': 'gravity'})
```

**Our code runs it** (`run_tool()`). The tool looks "gravity" up in
`tools/data/topic_facts.json`:

```
14:01:49 [tool]    get_topic_facts('gravity'): matched 'gravity', returning 6 facts
```

The notes go back to the model as a `ToolMessage`.

**Round 2.** With the notes in front of it, the model writes the quiz as
JSON:

```
14:01:49 [quiz_maker] calling qwen2.5:7b (round 2, 6 messages, tools: get_topic_facts)
14:03:12 [quiz_maker] reply received in 82.5s
14:03:12 [quiz_maker] final reply: 1078 characters
```

## Step 7: The quiz is checked and stored

`parse_quiz()` pulls the JSON out of the reply and validates it against the
`Quiz` pydantic class: 1–5 questions, 2–4 distinct choices each, and an
answer letter that matches a choice. Stray letters in choices ("A) Gravity")
are stripped. Then:

```
14:03:12 [quiz_maker] quiz ready: 3 questions on 'gravity'
```

The quiz goes into `state["quiz"]` as data, **not** into the chat, so the
answers stay hidden. The only chat message is `"I made a 3-question quiz on
gravity."`.

## Step 8: Handoff to the Tutor (`core/handoff.py`)

Nobody called the Quiz Maker (the return stack is empty), so it **transfers**
to the Tutor itself: its node returns
`"handoff": {"to": "tutor", "mode": "transfer"}`. The wrapper turns that into
`next_agent = "tutor"` and records a `handoff:quiz_maker->tutor` span:

```
14:03:12 [handoff] quiz_maker -> tutor (transfer)
```

`route_after_agent()` sees `next_agent` and sends the graph to `tutor`.
Checkpoint saved.

## Step 9: The Tutor presents the quiz (`agents/tutor.py`)

The wrapper clears `next_agent`. `tutor_node()` sees a quiz that has not been
presented yet, so it formats the questions **in plain Python**, with no model
call, and marks the quiz `presented`:

```
14:03:12 [tutor]   presenting the 3-question quiz on 'gravity'
```

`route_after_agent()` finds nothing more to do and returns `END`. The final
checkpoint is saved: 5 messages, plus the quiz.

## Step 10: Output (`chat.py`)

While all this ran, `chat.py` was reading the stream:

- **"messages" events** (model tokens): the router's and the Quiz Maker's
  were filtered out. Neither is chat for the student, and the Quiz Maker has
  `stream_replies=False`. The Tutor made no model call this time.
- **"updates" events** (a node finished): the Quiz Maker's and the Tutor's
  messages were printed as each node finished.

```
quiz_maker: I made a 3-question quiz on gravity.

tutor: Here's your quiz on gravity!

1. What is the force that causes objects with mass to attract each other?
   A) Magnetism
   B) Gravity
   ...
Reply with your answers, for example: 1-B, 2-A, 3-C
```

## Step 11: The trace is closed

The `chat_turn` span gets its outputs (the replies, `agents_involved:
[quiz_maker, tutor]`, the whole conversation) and closes. MLflow writes the
trace to `mlflow.db`:

```
14:03:12 [trace]   recorded trace tr-526fbe0883b4a069b07241d8b7ed2135
14:03:12 [chat]    saved: thread now has 5 messages
```

Where the time went: finding the topic 22 s (14:00:29 → 14:00:51), tool
round 1 59 s, quiz writing 83 s, plus the router, whose start time this log
does not show (10–40 s in other runs). Our own code took a few milliseconds
of the total. On a CPU, the time is almost all in the model.

### And the answer that followed

```
You: 1-B, 2-A, 3-C
14:05:02 [router]  -> tutor (The student is answering the quiz questions.)
14:05:02 [tutor]   graded the quiz in code: 2/3
```

The router picked the Tutor, and `tutor_node()` recognised quiz answers and
graded them against the stored key **in Python**, in under a millisecond.

---

## Example: a clarifying question

A new conversation, where the topic is missing. *(Log lines are what the code
prints; timings are left out.)*

```
You: make me a quiz
[router]     -> quiz_maker (...)
[quiz_maker] topic: (none found)
```

1. **The decision to ask is made in code.** `find_topic()` returned `""`, so
   `quiz_maker_node()` builds the reply
   `"CLARIFY: Which topic would you like the quiz to be about?"` and
   `clarification_update()` turns it into:
   ```python
   {"messages": [AIMessage("Which topic would you like the quiz to be about?")],
    "pending_question": {"agent": "quiz_maker", "question": "Which topic ..."}}
   ```
2. **Routing.** `route_after_agent()` sees `pending_question` and goes to the
   `ask_user` node.
3. **Pause.** `ask_user_node()` calls `interrupt(...)`. The run stops in the
   middle, and the checkpointer saves it, with `ask_user` as the next node:
   ```
   [clarify] waiting for the user to answer quiz_maker's question
   ```
   The stream ends. `chat.py` sees the thread is paused and prints:
   ```
   quiz_maker asks: Which topic would you like the quiz to be about?
     (paused - type your answer, or /quit and resume later)
   ```
   You could quit now; `python chat.py --thread <id>` would show the same
   question again.
4. **Resume.** You type `volcanoes`. Because the thread is paused,
   `graph_input()` sends `Command(resume="volcanoes")`, **not** a new
   message, so the router is skipped. LangGraph loads the paused checkpoint
   and runs `ask_user` again; this time `interrupt()` *returns* `"volcanoes"`:
   ```
   [clarify] resumed: user answered 'volcanoes', returning to quiz_maker
   ```
   `ask_user` adds your answer to the conversation, clears
   `pending_question`, and returns `Command(goto="quiz_maker")`.
5. **Carry on.** The Quiz Maker runs again; `find_topic()` now sees your
   answer:
   ```
   [quiz_maker] topic: 'volcanoes'
   [tool]       get_topic_facts('volcanoes'): no notes found
   ```
   The tool has no notes on volcanoes and says so, and the model writes basic
   questions from its own knowledge. Then the flow continues exactly as in
   Steps 7–10 above.

In MLflow this is **two traces**: the paused one (it ends at `ask_user`), and
the one for your answer, whose `chat_turn` has
`answers_clarifying_question = true` and contains a `clarification` span with
the question and your answer.

---

## Where to read more

| Topic | Doc | Code |
| --- | --- | --- |
| Agents | [docs/01_agents.md](docs/01_agents.md) | `agents/` |
| Router | [docs/02_router.md](docs/02_router.md) | `core/router.py` |
| Handoffs | [docs/03_handoffs.md](docs/03_handoffs.md) | `core/handoff.py` |
| Clarifying questions | [docs/04_clarifying_questions.md](docs/04_clarifying_questions.md) | `core/clarify.py` |
| Tool calling | [docs/05_tool_calling.md](docs/05_tool_calling.md) | `agents/base.py`, `tools/` |
| Checkpointing | [docs/06_checkpointing.md](docs/06_checkpointing.md) | `infra/postgres.py` |
| Tracing | [docs/07_tracing.md](docs/07_tracing.md) | `infra/tracing.py` |
