"""Non-wiki subagent tools: ToolSpecs + executors, one per action.

The wiki surface lives in :mod:`autoresearch.wiki.tool` (WIKI_READ_TOOLS /
WIKI_WRITE_TOOLS / WIKI_EXECUTORS). This module holds the rest — the tools that
touch the lab, the filesystem, the web, and the GPU queue — and follows the same
per-action shape: one tight ToolSpec and one small executor per tool. The runner
(:mod:`autoresearch.subagent.runner`) picks which of these an agent gets by type
and binds each executor to that agent's context (lab dir, allowed roots, the
injected experiment runner).

Write separation is enforced here, not just documented (DESIGN.md — "write tools
are strictly separated"):

- ``write`` resolves STRICTLY inside the current lab. It rejects absolute paths,
  ``..`` traversal, symlink escapes (a ``resolve()`` containment check), anything
  under the lab's ``runs/`` archive (past runs are immutable), and anything that
  resolves onto a pinned asset (dataset / tokenizer). It can produce nothing but a
  lab file — never a wiki file, never an archived run.
- ``read_file`` / ``list_dir`` are read-only over a configurable set of allowed
  roots (the lab dir and the labs/run archive) with the same traversal guards.

Descriptions load verbatim from ``prompt/tools/<name>.md``.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from autoresearch.llm.base import ToolCall, ToolSpec
from autoresearch.subagent.analyze import analyze

try:  # DuckDuckGo search is the same package research-bot uses (pyproject: ddgs)
    from ddgs import DDGS
except ImportError:  # keep the module importable where the extra isn't installed
    DDGS = None  # type: ignore[assignment]

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompt"

# Byte caps: read_file returns a head slice, fetch_page truncates readable text.
READ_FILE_CAP = 200_000
FETCH_CAP = 100_000
FETCH_TIMEOUT = 15.0
FETCH_MAX_BYTES = 5 * 1024 * 1024


@lru_cache(maxsize=None)
def _tool_description(name: str) -> str:
    """Load a tool's user-facing description verbatim from prompt/tools/<name>.md."""
    return (_PROMPT_DIR / "tools" / f"{name}.md").read_text(encoding="utf-8").strip()


def _spec(name: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return ToolSpec(name=name, description=_tool_description(name), input_schema=schema)


# ----- ToolSpecs -----

WRITE_TOOL = _spec(
    "write",
    {
        "path": {
            "type": "string",
            "description": "File path relative to your lab root (e.g. 'main.py').",
        },
        "content": {"type": "string", "description": "The full file contents to write."},
    },
    ["path", "content"],
)

READ_FILE_TOOL = _spec(
    "read_file",
    {
        "path": {
            "type": "string",
            "description": "File path relative to an allowed root (lab or archive).",
        }
    },
    ["path"],
)

LIST_DIR_TOOL = _spec(
    "list_dir",
    {
        "path": {
            "type": "string",
            "description": "Directory path relative to an allowed root; omit for the root.",
        }
    },
    [],
)

WEB_SEARCH_TOOL = _spec(
    "web_search",
    {
        "query": {"type": "string", "description": "The search query (topic or keywords)."},
        "max_results": {
            "type": "integer",
            "description": "Number of results to return (default 5).",
        },
    },
    ["query"],
)

FETCH_PAGE_TOOL = _spec(
    "fetch_page",
    {"url": {"type": "string", "description": "The URL of the page to fetch."}},
    ["url"],
)

ANALYZE_RECORDS_TOOL = _spec(
    "analyze_records",
    {
        "code": {
            "type": "string",
            "description": "A stdlib-only Python snippet; print your result to stdout.",
        },
        "cwd": {
            "type": "string",
            "description": (
                "Directory (relative to an allowed root) the snippet runs in, so paths "
                "like 'runs/3/records.jsonl' resolve. Defaults to the lab root."
            ),
        },
    },
    ["code"],
)

RUN_EXPERIMENT_TOOL = _spec("run_experiment", {}, [])


# ----- path guards -----

def _resolve_within(
    roots: Sequence[Path],
    rel: Any,
    *,
    forbid: Iterable[Path] = (),
) -> tuple[Path | None, str | None]:
    """Resolve ``rel`` under one of ``roots``, rejecting escape.

    Rejects absolute paths and ``..`` components up front, then resolves the
    candidate (following symlinks) and requires it to stay inside one of the
    allowed roots and outside every path in ``forbid``. Returns ``(path, None)``
    on success or ``(None, error)`` with a human-readable reason.
    """
    if not isinstance(rel, str) or not rel.strip():
        return None, "path is required and must be a non-empty string."
    rel = rel.strip()
    if rel.startswith("/") or Path(rel).is_absolute():
        return None, f"path must be relative to a lab/archive root, not absolute: {rel!r}"
    if ".." in Path(rel).parts:
        return None, f"path must not contain '..': {rel!r}"

    forbid_resolved = [f.resolve() for f in forbid]
    for root in roots:
        root_r = root.resolve()
        candidate = (root_r / rel).resolve()
        if candidate != root_r and root_r not in candidate.parents:
            continue  # escapes this root (e.g. via a symlink) — try the next
        for bad in forbid_resolved:
            if candidate == bad or bad in candidate.parents:
                return None, (
                    f"path resolves into a protected location and is refused: {rel!r}"
                )
        return candidate, None

    return None, f"path resolves outside every allowed root and is refused: {rel!r}"


# ----- executors -----

def exec_write(tool_call: ToolCall, lab_dir: Path, pinned_assets: Sequence[Path] = ()) -> str:
    """Write a file strictly inside ``lab_dir`` (executor-only)."""
    args = tool_call.arguments
    content = args.get("content")
    if not isinstance(content, str):
        return "content is required and must be a string."
    forbid = [lab_dir / "runs", *pinned_assets]
    target, error = _resolve_within([lab_dir], args.get("path"), forbid=forbid)
    if error:
        return error
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {target.relative_to(lab_dir.resolve())}."


def exec_read_file(tool_call: ToolCall, allowed_roots: Sequence[Path]) -> str:
    """Read one file from within ``allowed_roots`` (read-only)."""
    target, error = _resolve_within(allowed_roots, tool_call.arguments.get("path"))
    if error:
        return error
    assert target is not None
    if not target.exists():
        return f"No such file: {tool_call.arguments.get('path')!r}"
    if target.is_dir():
        return f"{tool_call.arguments.get('path')!r} is a directory; use list_dir."
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Could not read file: {e}"
    if len(text) > READ_FILE_CAP:
        text = text[:READ_FILE_CAP] + f"\n\n[truncated — {len(text)} chars total]"
    return text


def exec_list_dir(tool_call: ToolCall, allowed_roots: Sequence[Path]) -> str:
    """List a directory's entries from within ``allowed_roots`` (read-only)."""
    rel = tool_call.arguments.get("path")
    if rel is None or (isinstance(rel, str) and not rel.strip()):
        rel = "."
    target, error = _resolve_within(allowed_roots, rel)
    if error:
        return error
    assert target is not None
    if not target.exists():
        return f"No such directory: {rel!r}"
    if not target.is_dir():
        return f"{rel!r} is a file; use read_file."
    entries = []
    for child in sorted(target.iterdir(), key=lambda p: p.name):
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


async def exec_web_search(tool_call: ToolCall) -> str:
    """DuckDuckGo text search via the ``ddgs`` package (read-only)."""
    args = tool_call.arguments
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return "query is required and must be a non-empty string."
    max_results = args.get("max_results", 5)
    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5
    if DDGS is None:
        return "web_search is unavailable: the 'ddgs' package is not installed."
    try:
        results = await asyncio.get_running_loop().run_in_executor(
            None, lambda: DDGS().text(query, max_results=max_results)
        )
    except Exception as e:  # network / parsing failures must not crash the loop
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}" for r in results
    )


class _TextExtractor:
    """Minimal stdlib HTML-to-text extractor (skips script/style/nav/etc.)."""

    _SKIP = ("script", "style", "nav", "footer", "head")

    def __init__(self) -> None:
        from html.parser import HTMLParser

        outer = self

        class _P(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: list[str] = []
                self.skip = 0

            def handle_starttag(self, tag: str, attrs: Any) -> None:
                if tag in outer._SKIP:
                    self.skip += 1

            def handle_endtag(self, tag: str) -> None:
                if tag in outer._SKIP and self.skip > 0:
                    self.skip -= 1

            def handle_data(self, data: str) -> None:
                if self.skip == 0 and data.strip():
                    self.parts.append(data.strip())

        self._p = _P()

    def extract(self, html: str) -> str:
        self._p.feed(html)
        return "\n".join(self._p.parts)


async def _http_get(url: str) -> httpx.Response:
    """GET ``url`` following redirects under a timeout. Overridable in tests."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response


async def exec_fetch_page(tool_call: ToolCall) -> str:
    """Fetch a URL and return its readable text (read-only)."""
    url = tool_call.arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return "url is required and must be a non-empty string."
    try:
        response = await _http_get(url.strip())
    except Exception as e:  # any HTTP failure becomes a string, never a crash
        return f"Failed to fetch page: {e}"
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"Unsupported content type: {content_type or 'unknown'}"
    raw = response.text
    if len(raw.encode("utf-8", "ignore")) > FETCH_MAX_BYTES:
        return f"Page is too large to fetch (> {FETCH_MAX_BYTES:,} bytes)."
    text = _TextExtractor().extract(raw)
    if not text:
        return "No readable content found."
    if len(text) > FETCH_CAP:
        text = text[:FETCH_CAP] + f"\n\n[truncated — {len(text)} chars total]"
    return text


def exec_analyze_records(tool_call: ToolCall, allowed_roots: Sequence[Path]) -> str:
    """Run an inline stdlib Python snippet over run bookkeeping (read-only intent)."""
    args = tool_call.arguments
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return "code is required and must be a non-empty string."
    rel = args.get("cwd")
    if rel is None or (isinstance(rel, str) and not rel.strip()):
        cwd = allowed_roots[0].resolve()
    else:
        cwd, error = _resolve_within(allowed_roots, rel)
        if error:
            return error
        assert cwd is not None
    if not cwd.is_dir():
        return f"cwd is not a directory: {rel!r}"
    result = analyze(code, cwd=cwd)
    out = result.stdout.rstrip("\n")
    parts = [out] if out else []
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.rstrip(chr(10))}")
    status = "timed out" if result.timed_out else f"exit {result.returncode}"
    parts.append(f"[{status}]")
    return "\n".join(parts) if parts else f"[{status}]"


# The tool takes no arguments: the exact configuration comes solely from the lab's
# files (run_config.toml + code), which the worker snapshots and runs — never from
# tool-call args. The injected callable is already bound to the agent's lab context
# by the queue-worker integration, so it needs no logical arguments of its own.
RunExperimentCallable = Callable[[], "str | Awaitable[str]"]


async def exec_run_experiment(
    tool_call: ToolCall, run_callable: RunExperimentCallable | None
) -> str:
    """Submit an experiment through the injected runner (executor-only).

    Takes no arguments — the run is defined entirely by the current lab snapshot.
    The runner is wired in later by the queue-worker integration; without it this
    returns a clear, non-crashing error so a session can still finish cleanly.
    """
    if run_callable is None:
        return (
            "run_experiment is unavailable in this session: no experiment runner is "
            "wired up. Report this as a blocker in your summary."
        )
    try:
        result = run_callable()
        if inspect.isawaitable(result):
            result = await result
    except Exception as e:
        return f"run_experiment failed: {e}"
    return str(result)
