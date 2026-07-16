Mark a source in the shared research wiki as bad or superseded. Requires `id` and `reason`;
optional `superseded_by` points at the replacement source. Goes through the wiki store.

Sources are append-only: a source that turns out to be wrong is *retracted*, not removed.
The file stays on disk as history, but search stops returning it and read shows a
retraction banner. Use this instead of only noting a bad source in prose — a note in prose
does not stop search from retrieving the bad record. After retracting, run `wiki_audit` to
find summaries that relied on it.

Id format: lowercase with hyphens, no spaces (e.g. "arxiv-1706-03762").
