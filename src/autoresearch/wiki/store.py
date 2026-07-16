"""Two-tier wiki store, ported/adapted from ../research-bot/src/wiki/store.py.

Ground truth is markdown folders (shared across all goals):
    wiki-library/sources/<id>.md    immutable evidence (origin: url|capture|experiment);
                                    retract-not-delete, with reason + optional superseded_by
    wiki-library/summary/<slug>.md  editable synthesis; must cite via inline (source: id);
                                    writes citing unknown ids are REJECTED

Beside the files, a REBUILDABLE index (wiki-library/.index/): citations/backlinks
table + lexical FTS (stdlib sqlite3 FTS5) + semantic embeddings (chromadb extra).
Search runs both paths and appends a divergence warning when they disagree
(<50% overlap) — carried over from research-bot.

NEW versus research-bot (DESIGN.md — Knowledge base):
- typed notes:  type in {paper, mechanism, idea, experiment, result}
- typed links:  (extends|combines|refutes: <id-or-slug>) inline references,
  indexed like citations, so agents can query the idea graph.

Dropped from research-bot: the tamper-evidence layer (.snapshots/, selections.jsonl)
— its own devs removed it ("single operator, no adversary").

TODO(implement): port WikiStore, add the typed-relation extraction + graph queries,
and an `audit` action (citing-retracted / citing-missing / zero-citation notes).
"""
