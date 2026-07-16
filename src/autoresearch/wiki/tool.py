"""Wiki tool schemas + executors — one tool per action.

DESIGN.md ("Tools are per-action, and write tools are strictly separated") splits the
wiki surface into many small tools rather than two multiplexed ones. Each tool has a tight
per-action JSON schema and its own executor; the way the lab-scoped ``write`` tool stays
separate from the wiki writers is the same principle applied throughout.

- Read tools (:data:`WIKI_READ_TOOLS`) — ``wiki_search`` / ``wiki_read`` / ``wiki_list`` /
  ``wiki_graph_*`` … over both summaries and sources. They never mutate, so they are safe
  for either subagent type.
- Write tools (:data:`WIKI_WRITE_TOOLS`) — ``wiki_capture_source`` / ``wiki_write_summary``
  / ``wiki_retract_source``. These are the ONLY way to change the wiki, and every one goes
  through :class:`~autoresearch.wiki.store.WikiStore`, so the wiki's invariants (immutable
  sources, enforced citations, typed notes) always hold. They can produce nothing but wiki
  files.

The lab-scoped ``write`` tool is NOT here on purpose — the wiki write tools must never be
merged with it (see prompt/tools/write.md).

Every tool's user-facing description loads verbatim from prompt/tools/<name>.md, and
:data:`WIKI_EXECUTORS` maps each tool name to the executor that runs it.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from autoresearch.llm.base import ToolCall, ToolSpec
from autoresearch.wiki.store import NOTE_TYPES, RELATION_TYPES, WikiStore

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompt"


@lru_cache(maxsize=None)
def _tool_description(name: str) -> str:
    """Load a tool's user-facing description verbatim from prompt/tools/<name>.md."""
    return (_PROMPT_DIR / "tools" / f"{name}.md").read_text(encoding="utf-8").strip()


# ----- small arg helpers (no separate tool.args package in autoresearch) -----

def _required_str(args: dict, name: str) -> tuple[str | None, str | None]:
    value = args.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, f"{name} is required."
    if not isinstance(value, str):
        return None, f"{name} must be a string."
    return value.strip(), None


def _str_list(args: dict, name: str) -> tuple[list[str] | None, str | None]:
    value = args.get(name)
    if value is None:
        return None, None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None, f"{name} must be an array of strings."
    return value, None


def _spec(name: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    """Build a ToolSpec with a tight per-action schema (description from the prompt file)."""
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return ToolSpec(name=name, description=_tool_description(name), input_schema=schema)


# ----- reusable schema fragments -----

_SLUG = {"type": "string", "description": "Summary identifier (lowercase-with-hyphens)."}
_SOURCE_ID = {"type": "string", "description": "Source identifier (lowercase-with-hyphens)."}
_QUERY = {"type": "string", "description": "Search query (topic or keywords)."}
_N_RESULTS = {"type": "integer", "description": "Number of results to return (default 5)."}
_NOTE_TYPE = {"type": "string", "enum": list(NOTE_TYPES), "description": "Note type."}
_RELATION = {"type": "string", "enum": list(RELATION_TYPES), "description": "Relation type."}


# ----- read-tool schemas -----

WIKI_SEARCH_TOOL = _spec(
    "wiki_search",
    {"query": _QUERY, "n_results": _N_RESULTS},
    ["query"],
)

WIKI_READ_TOOL = _spec("wiki_read", {"slug": _SLUG}, ["slug"])

WIKI_LIST_TOOL = _spec(
    "wiki_list",
    {
        "type": {**_NOTE_TYPE, "description": "Filter to one note type."},
        "tag": {"type": "string", "description": "Filter to a single tag."},
    },
    [],
)

WIKI_INFO_TOOL = _spec("wiki_info", {"slug": _SLUG}, ["slug"])

WIKI_TAGS_TOOL = _spec("wiki_tags", {}, [])

WIKI_HISTORY_TOOL = _spec(
    "wiki_history",
    {"n": {"type": "integer", "description": "Number of recent timeline entries (default 10)."}},
    [],
)

WIKI_AUDIT_TOOL = _spec("wiki_audit", {}, [])

WIKI_SEARCH_SOURCES_TOOL = _spec(
    "wiki_search_sources",
    {"query": _QUERY, "n_results": _N_RESULTS},
    ["query"],
)

WIKI_READ_SOURCE_TOOL = _spec("wiki_read_source", {"id": _SOURCE_ID}, ["id"])

WIKI_LIST_SOURCES_TOOL = _spec(
    "wiki_list_sources",
    {"author": {"type": "string", "description": "Filter sources by author."}},
    [],
)

WIKI_SOURCE_INFO_TOOL = _spec("wiki_source_info", {"id": _SOURCE_ID}, ["id"])

WIKI_GRAPH_NEIGHBORS_TOOL = _spec(
    "wiki_graph_neighbors",
    {
        "slug": _SLUG,
        "relation": {**_RELATION, "description": "Optional relation to filter the edges to."},
    },
    ["slug"],
)

WIKI_GRAPH_EDGES_TOOL = _spec(
    "wiki_graph_edges",
    {"relation": {**_RELATION, "description": "Relation type to list every edge of."}},
    ["relation"],
)

WIKI_GRAPH_ORPHANS_TOOL = _spec(
    "wiki_graph_orphans",
    {
        "type": {
            "type": "string",
            "enum": [*NOTE_TYPES, "any"],
            "description": "Note type to scan (default 'idea'; 'any' for all).",
        }
    },
    [],
)


# ----- write-tool schemas -----

WIKI_CAPTURE_SOURCE_TOOL = _spec(
    "wiki_capture_source",
    {
        "id": _SOURCE_ID,
        "title": {"type": "string", "description": "Human-readable title."},
        "content": {"type": "string", "description": "The raw text to archive verbatim."},
        "url": {"type": "string", "description": "Origin URL of a fetched page (sets origin=url)."},
        "author": {"type": "string", "description": "Who wrote/said the captured text."},
    },
    ["id", "title", "content"],
)

WIKI_WRITE_SUMMARY_TOOL = _spec(
    "wiki_write_summary",
    {
        "slug": _SLUG,
        "title": {"type": "string", "description": "Human-readable title."},
        "type": {**_NOTE_TYPE, "description": "Note type (required)."},
        "content": {
            "type": "string",
            "description": (
                "The note body in markdown, with inline citations '(source: id)' and inline "
                "typed relations like '(extends: slug)' / '(combines: a, b)' / '(refutes: slug)'."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags.",
        },
    },
    ["slug", "title", "type", "content"],
)

WIKI_RETRACT_SOURCE_TOOL = _spec(
    "wiki_retract_source",
    {
        "id": _SOURCE_ID,
        "reason": {"type": "string", "description": "Why the source is being retracted."},
        "superseded_by": {
            "type": "string",
            "description": "Id of the source that replaces the retracted one (optional).",
        },
    },
    ["id", "reason"],
)


# ----- executors (one per tool; synchronous — wrap in asyncio.to_thread if needed) -----

def _exec_search(tool_call: ToolCall, store: WikiStore) -> str:
    query, error = _required_str(tool_call.arguments, "query")
    if error:
        return error
    return store.search(query, tool_call.arguments.get("n_results", 5))


def _exec_read(tool_call: ToolCall, store: WikiStore) -> str:
    slug, error = _required_str(tool_call.arguments, "slug")
    if error:
        return error
    return store.read(slug)


def _exec_list(tool_call: ToolCall, store: WikiStore) -> str:
    args = tool_call.arguments
    return store.list_notes(args.get("type"), args.get("tag"))


def _exec_info(tool_call: ToolCall, store: WikiStore) -> str:
    slug, error = _required_str(tool_call.arguments, "slug")
    if error:
        return error
    return store.note_info(slug)


def _exec_tags(tool_call: ToolCall, store: WikiStore) -> str:
    return store.list_tags()


def _exec_history(tool_call: ToolCall, store: WikiStore) -> str:
    return store.history(tool_call.arguments.get("n", 10))


def _exec_audit(tool_call: ToolCall, store: WikiStore) -> str:
    return store.audit()


def _exec_search_sources(tool_call: ToolCall, store: WikiStore) -> str:
    query, error = _required_str(tool_call.arguments, "query")
    if error:
        return error
    return store.search_sources(query, tool_call.arguments.get("n_results", 5))


def _exec_read_source(tool_call: ToolCall, store: WikiStore) -> str:
    source_id, error = _required_str(tool_call.arguments, "id")
    if error:
        return error
    return store.read_source(source_id)


def _exec_list_sources(tool_call: ToolCall, store: WikiStore) -> str:
    return store.list_sources(tool_call.arguments.get("author"))


def _exec_source_info(tool_call: ToolCall, store: WikiStore) -> str:
    source_id, error = _required_str(tool_call.arguments, "id")
    if error:
        return error
    return store.source_info(source_id)


def _exec_graph_neighbors(tool_call: ToolCall, store: WikiStore) -> str:
    slug, error = _required_str(tool_call.arguments, "slug")
    if error:
        return error
    return store.graph_neighbors(slug, tool_call.arguments.get("relation"))


def _exec_graph_edges(tool_call: ToolCall, store: WikiStore) -> str:
    relation, error = _required_str(tool_call.arguments, "relation")
    if error:
        return error
    return store.graph_edges(relation)


def _exec_graph_orphans(tool_call: ToolCall, store: WikiStore) -> str:
    return store.graph_orphans(tool_call.arguments.get("type"))


def _exec_capture_source(tool_call: ToolCall, store: WikiStore) -> str:
    args = tool_call.arguments
    source_id, error = _required_str(args, "id")
    if error:
        return error
    title, error = _required_str(args, "title")
    if error:
        return error
    content, error = _required_str(args, "content")
    if error:
        return error
    url = args.get("url")
    origin = "url" if url else "capture"
    return store.capture_source(source_id, title, content, origin, url, args.get("author"))


def _exec_write_summary(tool_call: ToolCall, store: WikiStore) -> str:
    args = tool_call.arguments
    slug, error = _required_str(args, "slug")
    if error:
        return error
    title, error = _required_str(args, "title")
    if error:
        return error
    note_type, error = _required_str(args, "type")
    if error:
        return error
    content = args.get("content", "")
    if not isinstance(content, str):
        return "content must be a string."
    tags, error = _str_list(args, "tags")
    if error:
        return error
    return store.write_summary(slug, title, note_type, content, tags)


def _exec_retract_source(tool_call: ToolCall, store: WikiStore) -> str:
    args = tool_call.arguments
    source_id, error = _required_str(args, "id")
    if error:
        return error
    reason, error = _required_str(args, "reason")
    if error:
        return error
    return store.retract_source(source_id, reason, args.get("superseded_by"))


# ----- tool groups + name->executor dispatch -----

WIKI_READ_TOOLS: list[ToolSpec] = [
    WIKI_SEARCH_TOOL,
    WIKI_READ_TOOL,
    WIKI_LIST_TOOL,
    WIKI_INFO_TOOL,
    WIKI_TAGS_TOOL,
    WIKI_HISTORY_TOOL,
    WIKI_AUDIT_TOOL,
    WIKI_SEARCH_SOURCES_TOOL,
    WIKI_READ_SOURCE_TOOL,
    WIKI_LIST_SOURCES_TOOL,
    WIKI_SOURCE_INFO_TOOL,
    WIKI_GRAPH_NEIGHBORS_TOOL,
    WIKI_GRAPH_EDGES_TOOL,
    WIKI_GRAPH_ORPHANS_TOOL,
]

WIKI_WRITE_TOOLS: list[ToolSpec] = [
    WIKI_CAPTURE_SOURCE_TOOL,
    WIKI_WRITE_SUMMARY_TOOL,
    WIKI_RETRACT_SOURCE_TOOL,
]

# name -> executor(tool_call, store) -> str. Read and write tools share one dispatch map;
# the two lists above are what gate a tool by subagent type.
WIKI_EXECUTORS: dict[str, Callable[[ToolCall, WikiStore], str]] = {
    "wiki_search": _exec_search,
    "wiki_read": _exec_read,
    "wiki_list": _exec_list,
    "wiki_info": _exec_info,
    "wiki_tags": _exec_tags,
    "wiki_history": _exec_history,
    "wiki_audit": _exec_audit,
    "wiki_search_sources": _exec_search_sources,
    "wiki_read_source": _exec_read_source,
    "wiki_list_sources": _exec_list_sources,
    "wiki_source_info": _exec_source_info,
    "wiki_graph_neighbors": _exec_graph_neighbors,
    "wiki_graph_edges": _exec_graph_edges,
    "wiki_graph_orphans": _exec_graph_orphans,
    "wiki_capture_source": _exec_capture_source,
    "wiki_write_summary": _exec_write_summary,
    "wiki_retract_source": _exec_retract_source,
}


def execute_wiki_tool(tool_call: ToolCall, store: WikiStore) -> str:
    """Dispatch a wiki tool call to its executor by name.

    Synchronous; wrap in ``asyncio.to_thread`` if the caller's loop needs it.
    """
    executor = WIKI_EXECUTORS.get(tool_call.name)
    if executor is None:
        return f"Unknown wiki tool: {tool_call.name}"
    return executor(tool_call, store)
