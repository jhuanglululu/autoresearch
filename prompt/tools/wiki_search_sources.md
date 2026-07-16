Search the shared research wiki's **sources** — the immutable evidence layer of fetched
pages, captured text, and experiment run records — by topic or keyword. Read-only.

Runs two independent retrieval paths, semantic (embeddings) and lexical (BM25), and warns
when they disagree; treat a disagreement warning as low confidence and read the candidates
rather than trusting the ranking. (If the semantic index is unavailable the search
silently runs lexical-only.) Retracted sources are excluded from the live results and
listed separately if they matched.

Sources are the only evidence; prefer fetched URLs (re-checkable) and experiment runs
(reproducible) over plain captures (testimony).

Id format: lowercase with hyphens, no spaces (e.g. "arxiv-1706-03762").
