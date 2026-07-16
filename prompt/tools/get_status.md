Get a quick status snapshot: which subagents are running right now, the GPU job queue
depth (pending / running / done counts), and cumulative token usage per model. Use it to
answer the operator's status questions and to check spend. Takes no arguments. This is
read-only and cheap — it makes no LLM or subagent calls.
