# Multi-Agent Study Helper

A small, heavily commented multi-agent chatbot that runs **entirely on your
own machine**. It exists to make these ideas concrete:

| Idea | What you will see in this project |
| --- | --- |
| **Multi-agent system** | A Tutor and a Quiz Maker that share one conversation and hand work to each other |
| **Router** | A step that picks the right agent for every message you send |
| **Handoffs** | "Call and return" (the Tutor asks the Quiz Maker for a quiz) and "transfer" (the Quiz Maker finishes and triggers the Tutor) |
| **Clarifying questions** | An agent asks a follow-up question, the conversation pauses, and it resumes with your answer, even after a restart |
| **Tool calling** | The Quiz Maker asks our Python code to run `get_topic_facts`, and uses the result |
| **Checkpointing** | Every conversation is saved to PostgreSQL; quit and resume any time |
| **Tracing** | MLflow records every agent, tool and model call, viewable in a web UI |

Everything is free and open source: Ollama, LangGraph, LangChain,
PostgreSQL, MLflow, Docker. **No paid API keys.**

```
You: what is photosynthesis?
tutor: Photosynthesis is a process used by plants, algae, and some bacteria to make their own food ...

You: help me build a quiz on gravity
quiz_maker: I made a 3-question quiz on gravity.
tutor: Here's your quiz on gravity!
1. What is the force that causes objects with mass to attract each other?
   A) Magnetism   B) Gravity   C) Electromagnetism   D) Centrifugal force
...
You: 1-B, 2-A, 3-C
tutor: Here are your results for the quiz on gravity: ... Score: 2/3
```

**New here?** Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md): it follows that quiz
request step by step through the router, agents, tool, checkpoint and trace.

---

## Contents

1. [Setup on Ubuntu](#1-setup-on-ubuntu)
2. [Running the chat](#2-running-the-chat)
3. [Viewing traces in MLflow](#3-viewing-traces-in-mlflow)
4. [Project structure](#4-project-structure)
5. [Concepts (docs/)](#5-concepts)
6. [Configuration](#6-configuration)
7. [Adding a third agent](#7-adding-a-third-agent)
8. [Troubleshooting](#8-troubleshooting)
9. [Design decisions](#9-design-decisions)

---

## 1. Setup on Ubuntu

Tested on Ubuntu 24.04 with Python 3.12, CPU only (no GPU needed).

### 1.1 Python

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
```

### 1.2 Ollama (runs the language model locally)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

The installer sets up a background service. If `ollama --version` works but
the chat cannot reach it, start it with `sudo systemctl start ollama` (or run
`ollama serve` in a spare terminal).

Then download the model (about 4.7 GB):

```bash
ollama pull qwen2.5:7b
```

| Model | Tool calling | Notes |
| --- | --- | --- |
| `qwen2.5:7b` | yes | **Recommended**: reliable tool calls and JSON |
| `llama3.1:8b` | yes | Works, but often writes tool calls as plain text (the code compensates) |
| `llama3.2:3b` | yes | Smaller and faster, lower quality |
| `llama2`, `gemma:2b`, old `phi` | **no** | The Quiz Maker will not work |

A 7–8B model wants about 8 GB of free RAM. On a CPU, expect 10–90 seconds
per model call.

### 1.3 Docker (runs PostgreSQL)

```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"
```

Log out and back in (or run `newgrp docker`) so you can use Docker without
`sudo`, then check it works:

```bash
docker run --rm hello-world
```

If this fails on a machine that used to have **Docker Desktop**, see
[Troubleshooting](#8-troubleshooting).

### 1.4 The Python environment

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run `source .venv/bin/activate` in every new terminal before using the
project. If you use conda, run `conda deactivate` first, so the right Python
is used.

### 1.5 Start PostgreSQL

```bash
docker compose up -d
docker compose ps          # wait until STATUS shows "healthy"
```

It keeps running in the background and restarts after a reboot. Stop it with
`docker compose down` (conversations are kept) or `docker compose down -v`
(conversations are deleted).

### 1.6 Check the model connection

```bash
python scripts/check_ollama.py
```

It checks four things: Ollama is reachable, the model is downloaded, a basic
chat works, and **tool calling** works. All four should say `[PASS]`.

---

## 2. Running the chat

```bash
python chat.py                    # start a new conversation
python chat.py --list             # list saved conversations
python chat.py --thread <id>      # resume one (the id is printed at the top of each chat)
python chat.py --quiet            # hide the flow log, show only the chat
```

Replies stream in word by word. Commands inside the chat:

| Command | What it does |
| --- | --- |
| `/help` | List the commands |
| `/new` | Start a new conversation (the current one stays saved) |
| `/threads` | List saved conversations |
| `/history` | Show this conversation so far |
| `/quiz` | Peek at the latest quiz **with** its answers |
| `/quit` | Leave (also `quit`, `exit`, Ctrl+C, Ctrl+D) |

Ctrl+C while a reply is being written abandons that reply but keeps the chat
open.

**Things to try:**

| Type | What happens |
| --- | --- |
| `what is photosynthesis?` | Router → Tutor answers |
| `help me build a quiz on gravity` | Router → Quiz Maker → tool → quiz → Tutor presents it |
| `1-B, 2-A, 3-C` | Router → Tutor grades it (in code, instantly) |
| `make it harder` | Router → Quiz Maker writes a harder quiz on the same topic |
| `explain it` (in a new chat) | The Tutor asks which topic and **pauses**; your answer resumes it |
| `make me a quiz` (in a new chat) | The Quiz Maker asks for a topic and pauses |

**The flow log.** Lines like `14:00:51 [quiz_maker] topic: 'gravity'` show
each internal step as it happens. Set `LOG_LEVEL=DEBUG` in `.env` to also see
the full text of every message sent to the model, or use `--quiet` to hide
the log.

`scripts/try_checkpoint.py` is the same chat without streaming
(`graph.invoke` instead of `graph.stream`), kept for comparison. It has one
extra command, `/checkpoints`, which lists every snapshot saved for the
conversation.

---

## 3. Viewing traces in MLflow

Recording happens automatically while you chat. To look at the traces, run
this in a **second terminal** from the project folder:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://127.0.0.1:5000>, click the **study-helper** experiment, then
**Traces**. Each message you sent is one trace; click it to see the tree of
spans: router decision, agents, tool calls, handoffs, each model call, with
inputs, outputs and timings. **Sessions** in the sidebar groups the traces of
one conversation. Details: [docs/07_tracing.md](docs/07_tracing.md).

---

## 4. Project structure

```
MultAgentChatbot/
├── chat.py                      The terminal chat (streaming) - start here
├── HOW_IT_WORKS.md              One message followed end to end
├── docs/                        Short explanations of each concept
│
├── agents/                      One file per agent
│   ├── base.py                  AgentSpec, the model call + tool-calling loop, run_agent()
│   ├── registry.py              The ONE list of all agents
│   ├── tutor.py                 Agent 1: explains, presents and grades quizzes
│   ├── quiz_maker.py            Agent 2: writes quizzes using its tool
│   └── prompts/                 System prompts, loaded at runtime (edit freely)
│       ├── tutor.md
│       ├── quiz_maker.md
│       ├── quiz_maker_topic.md  The Quiz Maker's topic-finding step
│       └── router.md
│
├── tools/
│   ├── topic_facts.py           get_topic_facts: schema, docstring, code
│   └── data/topic_facts.json    The study notes it looks up (6 topics)
│
├── core/                        The machinery, not specific to any agent
│   ├── config.py                Every setting, read once from .env
│   ├── llm.py                   The Ollama connection + health checks
│   ├── prompts.py               Loads the Markdown prompts
│   ├── state.py                 The conversation state the graph passes around
│   ├── router.py                Picks the agent for each new message
│   ├── handoff.py               Handoffs between agents + "where next?"
│   ├── clarify.py               Clarifying questions: pause, ask, resume
│   ├── graph.py                 Wires router, agents and ask_user into a LangGraph graph
│   └── logging_setup.py         The one-line-per-step flow log
│
├── infra/                       Connections to outside services
│   ├── postgres.py              The Postgres checkpointer + thread helpers
│   └── tracing.py               MLflow tracing setup
│
├── scripts/
│   ├── check_ollama.py          Checks the model connection and tool calling
│   └── try_checkpoint.py        The same chat without streaming, plus /checkpoints
│
├── docker-compose.yml           PostgreSQL (port 5433)
├── requirements.txt             Python dependencies, each one commented
└── .env.example                 Settings template - copy to .env
```

Every Python file starts with a docstring explaining what it does and why,
written for beginners.

---

## 5. Concepts

Each is a short read, with pointers into the code:

| Concept | Doc |
| --- | --- |
| What an agent is | [docs/01_agents.md](docs/01_agents.md) |
| What a router is | [docs/02_router.md](docs/02_router.md) |
| How handoffs work | [docs/03_handoffs.md](docs/03_handoffs.md) |
| How clarifying questions work | [docs/04_clarifying_questions.md](docs/04_clarifying_questions.md) |
| How tool calling works | [docs/05_tool_calling.md](docs/05_tool_calling.md) |
| What checkpointing does | [docs/06_checkpointing.md](docs/06_checkpointing.md) |
| What MLflow tracing shows | [docs/07_tracing.md](docs/07_tracing.md) |

The graph, in one picture (`core/graph.py`):

```
                          +--> END                  (reply finished)
                          |
START -> router -> agent -+--> another agent        (handoff)
                     ^    |
                     |    +--> ask_user -- interrupt(): PAUSE, saved to Postgres
                     |              |      ...you answer...
                     +--------------+      Command(resume=...): continue
```

---

## 6. Configuration

All settings live in `.env` (template: `.env.example`), read in one place,
`core/config.py`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Which local model all agents use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama listens |
| `OLLAMA_TEMPERATURE` | `0.2` | Creativity (the Quiz Maker uses 0.4, the router 0.0) |
| `OLLAMA_TIMEOUT` | `180` | Seconds to wait for a model reply |
| `POSTGRES_URI` | `postgresql://studyhelper:studyhelper@localhost:5433/studyhelper` | Matches `docker-compose.yml` |
| `MLFLOW_TRACKING_URI` | `mlflow.db` in the project folder | Where traces are stored |
| `MLFLOW_EXPERIMENT` | `study-helper` | The experiment name in the MLflow UI |
| `MLFLOW_ENABLED` | `true` | `false` turns tracing off |
| `LOG_LEVEL` | `INFO` | `DEBUG` (show messages sent to the model), `INFO`, `WARNING` |

Prompts are read once at startup: after editing a file in `agents/prompts/`,
restart the chat.

---

## 7. Adding a third agent

Say, a **Flashcards** agent:

1. **`agents/flashcards.py`**: a node function plus an `AgentSpec`. For a
   plain chat agent, the node is one line:
   ```python
   from agents.base import AgentSpec, run_agent

   def flashcards_node(state):
       return run_agent(AGENT, state)

   AGENT = AgentSpec(
       name="flashcards",
       description="Makes flashcards for memorising facts. Choose it when the "
                   "student asks for flashcards or help memorising.",
       prompt_file="flashcards.md",
       node=flashcards_node,
   )
   ```
2. **`agents/prompts/flashcards.md`**: its system prompt.
3. **One line in `agents/registry.py`**: add `flashcards.AGENT` to `_ALL`.

Nothing else changes. The graph adds a node for it, the router reads its
`description`, the Tutor's list of agents it can hand off to includes it,
it gets its own colour in the chat, and checkpointing and tracing cover it
automatically. Give it tools with `tools=(...)`, a function from `tools/`.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not reach Ollama` | `sudo systemctl start ollama`, or run `ollama serve` |
| `'<model>' is not installed` | `ollama pull <model>` (`check_ollama.py` prints the exact command) |
| `Could not connect to PostgreSQL` | `docker compose up -d`, then `docker compose ps` until "healthy" |
| `ModuleNotFoundError` | `source .venv/bin/activate` (and `conda deactivate` first) |
| VS Code: "package not installed" hints | Ctrl+Shift+P → *Python: Select Interpreter* → `.venv/bin/python` |
| Replies time out | Raise `OLLAMA_TIMEOUT` in `.env`; the first call after startup loads the model and is slow |
| The Quiz Maker says it couldn't make a quiz | The model produced invalid JSON twice; try again, or use `qwen2.5:7b` |
| `[quiz_maker] model wrote the ... call as text; converting` | A `llama3.1` quirk, handled automatically; `qwen2.5:7b` avoids it |

**Docker after Docker Desktop.** If `docker run hello-world` says
*Cannot connect to the Docker daemon at unix:///home/<you>/.docker/desktop/docker.sock*,
the `docker` command still points at a removed Docker Desktop:

```bash
docker context use default              # point docker at Docker Engine
sudo apt purge -y docker-desktop        # remove its leftovers
sudo dpkg --configure -a                # finish any half-done install
sudo systemctl enable --now containerd docker
```

If `docker run` then says *docker-credential-desktop: executable file not found*,
open `~/.docker/config.json` and delete the line `"credsStore": "desktop",`.

---

## 9. Design decisions

**Code does what code can; the model does the language.** Everything that
needs understanding what the student means is done by a model: the router
choosing an agent for each message, the Tutor explaining, the Quiz Maker
reading the topic, calling its tool and writing questions. But small local
models follow strict rules poorly, so the purely mechanical jobs *inside*
the agents are done in Python: presenting a quiz, grading answers against the
key, stripping letters from choices, and the yes/no "is the topic missing?"
check. Each of those started as a model instruction and became reliable only
in code. See [docs/01_agents.md](docs/01_agents.md).

**Structured output for decisions.** The router's choice and the Quiz
Maker's topic come back as JSON constrained by a schema (Ollama's `format=`),
so a decision cannot drift into free text.

**The quiz is data, not a chat message.** It lives in `state["quiz"]` with
its answer key, so the student never sees the answers, while the Tutor can
still grade.

**One registry, no agent names in the machinery.** The router, graph and
handoff code read `agents/registry.py`; none of them mention a specific
agent.

**Settings live in one file.** `core/config.py` is the only module that
reads environment variables.

**Postgres on port 5433.** Avoids clashing with a PostgreSQL already
installed on 5432.

**MLflow is not in Docker.** It is a Python program writing to one SQLite
file, `mlflow.db`, which is one less container to manage.
