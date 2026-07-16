List the entries of a directory within your allowed roots (your lab and the
archive of past runs). Read-only.

The path is resolved against those roots and must stay inside them: absolute
paths, `..` traversal, and symlinks that escape are rejected. Directories are
shown with a trailing `/`. Omit `path` to list the root itself.

Input:
- `path` (optional): directory path relative to an allowed root
  (e.g. `runs`, `runs/3`). Defaults to the root.

Use this to find your way around a lab or the run archive, then `read_file` an
individual file.
