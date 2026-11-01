# How tool calling works

A language model can only produce text. A **tool** lets it use a Python
function: here, `get_topic_facts`, which returns study notes for a topic
from `tools/data/topic_facts.json`.

The key idea: **the model never runs the tool.** It *asks* for it, and our
code runs it.

## What the model sees

Not the Python code, only the tool's **schema**, built from
`tools/topic_facts.py`:

| Schema part | Comes from |
| --- | --- |
| name: `get_topic_facts` | `@tool("get_topic_facts", ...)` |
| description | the function's docstring |
| arguments: `topic` (string) + its description | the `TopicFactsInput` pydantic class |

So the docstring and field descriptions are not just comments: they tell the
model when and how to use the tool.

## The loop

The full loop is in `call_model()` in `agents/base.py`, with each step
numbered in the comments:

```
1. model_with_tools = model.bind_tools([get_topic_facts])
   Send the conversation; the tool schema goes along with it.

2. The model replies. Either
   a) plain text                          -> the final answer, stop
   b) reply.tool_calls =
      [{"name": "get_topic_facts",
        "args": {"topic": "gravity"},
        "id": "call_1"}]                   -> a REQUEST, nothing has run

3. Our code runs it:  run_tool() -> get_topic_facts.invoke({"topic": "gravity"})

4. The result goes back as a ToolMessage(content=<notes>, tool_call_id="call_1"),
   added to the conversation after the model's request.

5. Back to step 1. The model now writes the quiz using the notes.
```

From a real run:

```
[quiz_maker] calling qwen2.5:7b (round 1, 4 messages, tools: get_topic_facts)
[quiz_maker] model requested tool get_topic_facts({'topic': 'gravity'})
[tool]       get_topic_facts('gravity'): matched 'gravity', returning 6 facts
[quiz_maker] calling qwen2.5:7b (round 2, 6 messages, tools: get_topic_facts)
[quiz_maker] quiz ready: 3 questions on 'gravity'
```

## Safety details

- **Errors go back to the model as text**, not as crashes: an unknown tool
  name or bad arguments become a message the model can react to.
- **At most 3 rounds.** After that, the model must answer without tools.
- **Tool calls written as text.** `llama3.1:8b` sometimes writes
  `{"name": "get_topic_facts", "parameters": {...}}` as plain text instead
  of a real tool call. `_recover_text_tool_call()` converts it back.
  `qwen2.5:7b` rarely does this, which is why it is the recommended model.
- The tool requests and results stay **inside the agent**: only its final
  reply enters the shared conversation. The trace still shows every round.
