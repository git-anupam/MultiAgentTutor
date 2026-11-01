# How clarifying questions work

Sometimes an agent cannot do its job yet: "explain it" (explain *what*?),
"make me a quiz" (on *what* topic?). Instead of guessing, it asks the
student a question, **pauses**, and **resumes** with the answer. The code is
in `core/clarify.py`.

## The flow

```
You: explain it

1. tutor       the model replies "CLARIFY: Which topic would you like me to explain?"
               run_agent() spots the marker and sets
               pending_question = {"agent": "tutor", "question": "Which topic ...?"}

2. routing     route_after_agent() sees pending_question -> ask_user

3. ask_user    calls LangGraph's interrupt(...)
               The run STOPS. The checkpointer saves it to Postgres.
               The chat prints:  tutor asks: Which topic would you like me to explain?

You: photosynthesis

4. resume      the chat sees the thread is paused and sends
               Command(resume="photosynthesis") instead of a new message
               (no router: the paused run already knows who asked)

5. ask_user    runs again; this time interrupt() RETURNS "photosynthesis".
               It adds the answer to the conversation and returns
               Command(goto="tutor")

6. tutor       sees "explain it" / "Which topic?" / "photosynthesis",
               and answers properly
```

Because the paused run is saved in Postgres, you can **quit while a question
is waiting**, come back with `python chat.py --thread <id>`, and the chat
reminds you of the question. Your next line answers it.

## Why `ask_user` is a separate node

When a paused graph resumes, **the node that paused runs again from its
first line**, and this time `interrupt()` returns the answer instead of
stopping.

If the agent called the model and then paused in the same node, resuming
would call the model a second time. Its new reply might not ask anything,
and your answer would be lost. Keeping the model call (agent node) apart
from the pause (`ask_user` node) makes the re-run harmless: `ask_user` only
reads the question from the state.

## Two ways to decide to ask

| Agent | Who decides | How |
| --- | --- | --- |
| Tutor | The **model** | Replies with the `CLARIFY:` marker when a request is vague |
| Quiz Maker | **Code** | A small model call extracts the topic as `{"topic": "..."}`; `if not topic:` the code asks |

The Quiz Maker switched to the second style because the model kept asking
"which topic?" even for "make me a quiz on gravity". Extracting one value is
much easier for a small model than judging whether to ask. From step 2
onward, both styles use the same pause-and-resume machinery.
