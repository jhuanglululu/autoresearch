Read a single file's text from within your allowed roots (your lab and the
archive of past runs). Read-only.

The path is resolved against those roots and must stay inside them: absolute
paths, `..` traversal, and symlinks that escape are rejected. This tool reads
lab and archive files — it does NOT read the wiki (use `wiki_read` /
`wiki_read_source`) and cannot write anything.

Input:
- `path` (required): file path relative to an allowed root
  (e.g. `main.py`, `runs/3/record.md`).

Large files are truncated to a byte cap; read the part you need and use
`analyze_records` when a question spans many run logs.
