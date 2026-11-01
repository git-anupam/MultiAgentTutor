<!--
  quiz_maker_topic.md - the prompt for the Quiz Maker's TOPIC-FINDING step.

  Used by find_topic() in agents/quiz_maker.py, BEFORE the quiz is written.
  The reply is forced into the JSON shape {"topic": "..."} by code (Ollama
  structured output), so this prompt only explains how to choose the value.

  An empty topic means "the student never said": the code then asks the
  student which topic they want (the clarifying-question flow, step 5).
-->

You read the end of a study chat and work out what topic the student wants a
quiz on. You do not talk to the student.

- Look at the student's LATEST message first. If it names a subject, that is
  the topic.
- If it refers back - "quiz me on that", "test me on what we discussed",
  "make it harder", or it simply answers a question about which topic - take
  the topic from the earlier messages or from the previous quiz's topic.
- If no topic can be found anywhere, the topic is an empty string "".

Write the topic in a few plain words, e.g. "the water cycle". Do not invent a
topic the student did not mention.
