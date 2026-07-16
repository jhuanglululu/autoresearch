Kill a currently-running subagent session. Use this when the operator tells you to stop
or redirect one mid-run, or when you have decided its work is no longer worth finishing.

- `session_id`: the running session to kill (e.g. `exec-1`).

Cancels the subagent's task immediately. It is recorded as `killed`; its context is not
reusable afterward (you cannot `follow_up` a killed session), so if you still want its
work, spawn a fresh subagent with an amended prompt.
