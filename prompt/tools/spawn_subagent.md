Spawn a one-shot subagent to do work you must never do yourself (reading, searching,
coding, running experiments). You compose its entire instruction in `prompt`: the role
(brainstormer / literature scout / idea synthesizer / experiment engineer / utility
lookup), the concrete task, the detailed rules, and what belongs in the wiki/lab versus
the short summary it returns.

- `type`: `executor` (writes + runs experiment code, submits GPU jobs, writes wiki) or
  `researcher` (searches, reads, writes wiki notes — cannot edit or run code).
- `model`: the name of a configured subagent model. Pick per the task using the model
  descriptions in your system prompt.
- `lab_id`: REQUIRED for `executor`. Names the lab to work in; created from the goal
  baseline if new, reused if it already exists. Ignored for `researcher`.
- `timeout_s`: optional per-spawn wall-clock limit.

The subagent runs fire-and-forget: this call returns immediately with "running". After
it, reply to the operator and END your turn — you will get a fresh turn opened with the
subagent's summary (or its failure) when it finishes. Do not try to wait for it in the
same turn. Assign architecture-level ideas only, never trivia like "test a wider
d_model" — the engineer varies its own training dynamics.
