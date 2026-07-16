Ask a FINISHED subagent session a follow-up question with its full context intact.
Prefer this over spawning a new subagent whenever the answer depends on what a previous
one already did or read — respawning would force a fresh agent to rediscover the same
context and burn tokens.

- `session_id`: a finished session id (e.g. `exec-1`, `res-2`) — see `list_sessions`.
- `question`: what to ask it.

Returns the session's answer. Only works on sessions that finished normally (done,
failed, or timed out); a killed session's context is gone — respawn instead. The
follow-up runs to completion before your turn continues.
