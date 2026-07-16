Search the shared research wiki's **summaries** — the curated, typed notes that distil
knowledge and cite sources — by topic or keyword. Read-only.

Runs two independent retrieval paths, semantic (embeddings) and lexical (BM25), and warns
when they disagree; treat a disagreement warning as low confidence and read the candidates
rather than trusting the ranking. (If the semantic index is unavailable the search
silently runs lexical-only.)

Only sources are evidence. A summary citing another summary is navigation, not support.
Use search to find a summary's slug, then read it with `wiki_read`.

Slug format: lowercase with hyphens, no spaces (e.g. "rope-scaling"). Note types are
paper / mechanism / idea / experiment / result.
