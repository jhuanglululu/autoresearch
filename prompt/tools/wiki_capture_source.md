Archive raw material into the shared research wiki as an immutable **source** — the only
evidence layer. Goes through the wiki store, which maintains the wiki's structure and
invariants; it cannot write arbitrary files. Use `write` for lab code — a separate,
lab-scoped tool that can NOT touch the wiki.

Sources are the unedited raw material: fetched pages, captured text, experiment run
records. Write-once — a source cannot be edited or deleted, so capture it verbatim and give
it a stable, fresh id. Fetch/read the page with your web tools first, then archive the text
here.

Requires `id`, `title`, `content`. Pass `url` for a fetched page (stored as a re-verifiable
origin) or omit it for a plain capture (testimony); optional `author` records who wrote or
said the captured text.

A source that turns out to be wrong is *retracted*, not removed — see `wiki_retract_source`.

Id format: lowercase with hyphens, no spaces (e.g. "arxiv-1706-03762").
