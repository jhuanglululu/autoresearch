Run a short inline Python snippet to analyze run bookkeeping — the per-run
`records.jsonl` (append-only event log) and `metrics.json` summaries under the
archive (`runs/`) — with READ-ONLY intent. Use this to answer questions across
runs that a single file read can't: compare the loss curves of runs 3 and 7,
find the step where val loss stopped improving, tabulate tokens/sec by model
size, and so on.

Input:
- `code` (required): a Python snippet, run as-is by a fresh stdlib-only
  interpreter. Print your result to stdout — stdout is the answer you get back.
- `cwd` (optional): directory the snippet runs in, so relative paths like
  `runs/3/records.jsonl` resolve. Defaults to the current location.

How it runs and its limits:
- Stdlib only (json, pathlib, statistics, ...). No third-party packages, no
  network — treat it as offline, read-only analysis. Do not try to install or
  import external libraries.
- Wall-clock timeout (default ~30s): an infinite loop or a huge scan is killed
  and reported as timed out. stdout/stderr are captured and truncated to a cap,
  so print summaries, not whole files.
- READ-ONLY is the contract, not a lock: the snippet is isolated but the
  filesystem is not hard-enforced read-only, so never write, move, or delete
  anything — read and compute only.

Return: the snippet's stdout (your result), plus stderr and exit status for
debugging a snippet that errors or times out.
