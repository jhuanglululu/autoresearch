"""One-shot subagent loop: initial prompt in -> tool calls -> summary out.

Toolsets by type (DESIGN.md). Wiki access is per-action, split into two groups:
WIKI_READ_TOOLS (wiki_search / wiki_read / wiki_list / wiki_info / wiki_tags /
wiki_history / wiki_audit / wiki_search_sources / wiki_read_source / wiki_list_sources /
wiki_source_info / wiki_graph_neighbors / wiki_graph_edges / wiki_graph_orphans — read-only,
safe for either type) and WIKI_WRITE_TOOLS (wiki_capture_source / wiki_write_summary /
wiki_retract_source — STRICTLY wiki-scoped, through the store so structure/invariants hold).

- executor:   WIKI_READ_TOOLS + WIKI_WRITE_TOOLS, plus its lab tools: `write` (STRICTLY
              lab-scoped: resolves inside the current lab, never wiki, never past runs/
              dirs, never dataset/tokenizer), archive read, `run_experiment` (the ONLY
              way to launch a run — no CLI, no env vars).
- researcher: web/arXiv search, page fetch, WIKI_READ_TOOLS + WIKI_WRITE_TOOLS, archive
              read. No `write`, no code execution.

The wiki write tools and the lab-scoped `write` are separate on purpose and must never be
merged: `write` cannot produce a wiki file, the wiki writers cannot produce a lab file.

Details are written to the wiki/lab; the final message returned to the
orchestrator is a short summary. Tool descriptions load verbatim from
prompt/tools/<name>.md.

TODO(implement): tool registry (with the path guards above) + async loop over
llm.base.LLMClient.
"""
