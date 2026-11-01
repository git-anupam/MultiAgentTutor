# What MLflow tracing shows

A **trace** is a detailed record of one message: a tree of **spans**, one
per piece of work, each with its inputs, outputs, duration and labels.
MLflow stores them in `mlflow.db` (SQLite) and shows them in a web page.
Setup: `infra/tracing.py`.

## Opening the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://127.0.0.1:5000> → **study-helper** → **Traces**. Click a trace.

## One trace = one message

Each message you send is one trace. A conversation is several traces that
share a thread id, and you can see them together in two ways:

- **Sessions** (left sidebar): each conversation, with its traces in order.
- The `conversation` output of the top `chat_turn` span: the whole chat so
  far, each line labelled with the agent that wrote it.

## The span tree for a quiz request

```
chat_turn                        your message, replies, agents involved, conversation
└── LangGraph
    ├── router
    │   └── route                chosen agent + reason
    │       └── ChatOllama
    ├── quiz_maker
    │   ├── quiz_maker:find_topic    topic: "gravity"
    │   ├── agent:quiz_maker         agent=quiz_maker, tools=[get_topic_facts]
    │   │   ├── ChatOllama           the model asks for the tool
    │   │   ├── tool:get_topic_facts input: topic, output: the notes
    │   │   └── ChatOllama           the model writes the quiz
    │   └── handoff:quiz_maker->tutor    mode=transfer
    └── tutor                    presents the quiz (no model call)
```

When you answer, the next trace shows `router → tutor → tutor:grade_quiz`,
with no `ChatOllama` under the Tutor, because grading is done in code.

## Where the spans come from

| Source | Spans |
| --- | --- |
| **Automatic**, `mlflow.langchain.autolog()` | `LangGraph`, one per node (`router`, `tutor`, ...), every `ChatOllama` call |
| **Our own**, `mlflow.start_span(...)` | `chat_turn`, `agent:<name>`, `tool:<name>`, `route`, `handoff:<a>-><b>`, `clarification`, `quiz_maker:find_topic`, `tutor:grade_quiz` |

The two kinds nest into one tree. Our spans add what autologging cannot
know: **which agent** did the work, the router's reason, the handoff mode.

## Useful things to look at

- A `ChatOllama` span's **inputs**: exactly what the model received,
  including the hidden system prompt.
- **Timing**: nearly all the time is spent in `ChatOllama`; the code itself
  takes milliseconds.
- **Tokens**: shown next to each `ChatOllama` span; they grow as the
  conversation gets longer.
- **Filter one conversation**: `tags.thread_id = 'chat-...'` in the search box.
- The **Pretty** view shows lists in YAML (`- >-` marks a multi-line item);
  switch to **JSON** for plain data.

Set `MLFLOW_ENABLED=false` in `.env` to turn tracing off.
