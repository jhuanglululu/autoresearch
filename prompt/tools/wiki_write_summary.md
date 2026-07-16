Create or update a **summary** in the shared research wiki — your own synthesis in clean
markdown. Goes through the wiki store, which enforces the wiki's invariants (immutable
sources, enforced citations, typed notes); it cannot write arbitrary files. Use `write`
for lab code — a separate, lab-scoped tool that can NOT touch the wiki.

Distil, don't transcribe. Requires `slug`, `title`, `type`, `content`; optional `tags`.

- `type` is one of: paper | mechanism | idea | experiment | result.
- **Only sources are evidence. A summary citing another summary is navigation, not support.**
  Cite evidence INLINE in the content as "(source: id)". A citation to an unknown source id
  — or to a summary slug — REJECTS the write. Capture the source first, then cite it.
- Link related notes INLINE with typed relations: "(extends: slug)", "(combines: a, b)",
  "(refutes: slug)". Targets may be summaries or sources; targets that don't exist yet are
  ignored with a warning (create them, then re-save).
- Prefer updating an existing summary over creating a duplicate on the same topic.

Slug format: lowercase with hyphens, no spaces (e.g. "rope-scaling").
