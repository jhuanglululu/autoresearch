Write to the shared research wiki — and nothing else. This tool goes through the
wiki store, which maintains the wiki's structure and invariants; it cannot write
arbitrary files (use write for lab code — a separate, lab-scoped tool).

Actions:
- capture_source: store immutable raw material (fetched page, capture). Sources are
  never edited or deleted afterwards.
- write_summary: create/update a summary note. Must cite evidence inline
  (source: id); writes citing unknown ids are REJECTED — capture sources first.
  Give the note a type (paper|mechanism|idea|experiment|result) and link related
  notes with typed relations (extends|combines|refutes).
- retract_source: exclude a bad source from search with a reason (file remains).
