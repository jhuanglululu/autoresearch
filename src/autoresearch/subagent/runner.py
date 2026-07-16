"""One-shot subagent loop: initial prompt in -> tool calls -> summary out.

Toolsets by type (DESIGN.md):
- executor:   `write` (STRICTLY lab-scoped: resolves inside the current lab, never
              wiki, never past runs/ dirs, never dataset/tokenizer), `wiki` (read/
              search), `wiki_write` (STRICTLY wiki-scoped, through the store so
              structure/invariants hold), archive read, `run_experiment` (the ONLY
              way to launch a run — no CLI, no env vars).
- researcher: web/arXiv search, page fetch, `wiki` + `wiki_write`, archive read.
              No `write`, no code execution.

The two write tools are separate on purpose and must never be merged: `write`
cannot produce a wiki file, `wiki_write` cannot produce a lab file.

Details are written to the wiki/lab; the final message returned to the
orchestrator is a short summary. Tool descriptions load verbatim from
prompt/tools/<name>.md.

TODO(implement): tool registry (with the path guards above) + async loop over
llm.base.LLMClient.
"""
