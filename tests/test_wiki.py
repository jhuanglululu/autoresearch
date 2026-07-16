"""Tests for the wiki subsystem (store + tools).

chromadb is an optional extra and is expected to be absent here; these tests
therefore exercise the graceful lexical-only degradation path. Every assertion is
written to hold on either path (with or without the semantic index).
"""
from __future__ import annotations

import json

import pytest

from pathlib import Path

from autoresearch.wiki.store import WikiStore
from autoresearch.wiki.tool import (
    WIKI_EXECUTORS,
    WIKI_READ_TOOLS,
    WIKI_WRITE_TOOLS,
    execute_wiki_tool,
)
from autoresearch.llm.base import ToolCall


@pytest.fixture
def store(tmp_path):
    return WikiStore(tmp_path / "wiki-library")


def _call(name, **args):
    return ToolCall(id="t", name=name, arguments=args)


# ----- source capture + immutability -----

def test_source_capture_and_immutability(store, tmp_path):
    msg = store.capture_source("arxiv-1", "Attention Paper", "Self-attention is all you need.", "url", url="https://x/1")
    assert "saved as sources/arxiv-1.md" in msg
    src_file = tmp_path / "wiki-library" / "sources" / "arxiv-1.md"
    assert src_file.exists()
    assert "Self-attention is all you need." in src_file.read_text()

    # Re-capture with same id is refused; original content is untouched.
    msg2 = store.capture_source("arxiv-1", "Different", "Overwrite attempt.", "capture")
    assert "immutable" in msg2.lower()
    assert "Overwrite attempt." not in src_file.read_text()

    # Empty content refused.
    assert "empty" in store.capture_source("empty-src", "T", "   ", "capture").lower()


# ----- summary writes: valid + invalid citations -----

def test_summary_write_valid_citation(store):
    store.capture_source("src-a", "Source A", "Rotary embeddings extend context.", "url", url="https://x/a")
    msg = store.write_summary(
        "rope-note", "RoPE note", "mechanism",
        "Rotary position embeddings help length generalization (source: src-a).",
    )
    assert "saved as summary/rope-note.md" in msg
    assert "unanchored" not in msg  # it has a citation
    body = store.read("rope-note")
    assert "Evidence: cites 1 source(s)" in body
    assert "externally re-verifiable" in body  # origin url


def test_summary_write_unknown_citation_rejected(store, tmp_path):
    msg = store.write_summary(
        "bad-note", "Bad", "idea",
        "This claims something (source: does-not-exist).",
    )
    assert "Not saved" in msg
    assert "does-not-exist" in msg
    assert not (tmp_path / "wiki-library" / "summary" / "bad-note.md").exists()


def test_summary_citing_summary_is_rejected(store):
    store.capture_source("src-x", "X", "content", "capture")
    store.write_summary("note-one", "One", "idea", "First note (source: src-x).")
    # Citing another *summary* as a source must be rejected — summaries are not evidence.
    msg = store.write_summary("note-two", "Two", "idea", "Builds on (source: note-one).")
    assert "Not saved" in msg
    assert "not sources" in msg


def test_zero_citation_warns_and_audit_flags(store):
    msg = store.write_summary("hunch", "Hunch", "idea", "Pure speculation, no evidence.")
    assert "saved" in msg
    assert "unanchored" in msg
    audit = store.audit()
    assert "hunch" in audit
    assert "unanchored" in audit


# ----- retraction + banner + audit -----

def test_retraction_banner_and_audit(store):
    store.capture_source("shaky", "Shaky", "A claim that will be retracted.", "capture")
    store.write_summary("relies", "Relies", "idea", "Depends on (source: shaky).")
    assert "Audit clean" in store.audit()

    msg = store.capture_source("solid", "Solid", "Replacement.", "url", url="https://x/solid")
    ret = store.retract_source("shaky", "Measurement error", superseded_by="solid")
    assert "retracted" in ret.lower()

    # read_source shows a banner.
    src_view = store.read_source("shaky")
    assert "RETRACTED" in src_view
    assert "Measurement error" in src_view
    assert "superseded_by" in src_view.lower() or "Superseded by" in src_view

    # Lexical search no longer returns the retracted source among live results.
    res = store.search_sources("retracted claim")
    live = res.split("Excluded")[0]
    assert "shaky" not in live

    # audit now flags the summary citing a retracted source.
    audit = store.audit()
    assert "relies" in audit
    assert "retracted" in audit

    # read of the summary shows the retracted-citation warning.
    assert "RETRACTED source" in store.read("relies")

    # Double retraction refused.
    assert "already retracted" in store.retract_source("shaky", "again")


# ----- typed relations + graph queries -----

def test_typed_relations_and_graph(store):
    store.capture_source("paper-1", "Paper 1", "base mechanism", "url", url="https://x/1")
    store.write_summary("mech-a", "Mechanism A", "mechanism", "Base mechanism (source: paper-1).")
    store.write_summary("mech-b", "Mechanism B", "mechanism", "Another base (source: paper-1).")
    store.write_summary(
        "idea-combo", "Combined idea", "idea",
        "Combine them (combines: mech-a, mech-b) and note it (extends: mech-a). (source: paper-1)",
    )
    store.write_summary(
        "idea-counter", "Counter idea", "idea",
        "This challenges the combo (refutes: idea-combo). (source: paper-1)",
    )
    # An unconnected idea.
    store.write_summary("lonely", "Lonely idea", "idea", "Nobody links me. (source: paper-1)")

    neigh = store.graph_neighbors("idea-combo")
    assert "combines → `mech-a`" in neigh
    assert "combines → `mech-b`" in neigh
    assert "extends → `mech-a`" in neigh
    assert "idea-counter" in neigh  # incoming refutes

    combos = store.graph_edges("combines")
    assert "`idea-combo` combines `mech-a`" in combos
    assert "`idea-combo` combines `mech-b`" in combos

    refutes = store.graph_edges("refutes")
    assert "`idea-counter` refutes `idea-combo`" in refutes

    orphans = store.graph_orphans()  # default: idea notes
    assert "lonely" in orphans
    assert "idea-combo" not in orphans  # it is connected

    # Filtering neighbours by relation.
    only_ext = store.graph_neighbors("idea-combo", relation="extends")
    assert "extends → `mech-a`" in only_ext
    assert "combines" not in only_ext


def test_relation_to_unknown_target_is_dropped_with_warning(store):
    store.capture_source("s1", "S1", "x", "capture")
    msg = store.write_summary("n1", "N1", "idea", "Extends nothing real (extends: ghost). (source: s1)")
    assert "ignored" in msg
    assert "ghost" in msg
    assert "No 'extends' edges" in store.graph_edges("extends")


# ----- FTS lexical search -----

def test_fts_search_finds_notes(store):
    store.capture_source("s-transformer", "Transformers", "The transformer architecture uses attention.", "url", url="https://x/t")
    store.write_summary(
        "transformer-note", "Transformer summary", "paper",
        "Transformers rely on multi-head attention mechanisms (source: s-transformer).",
    )
    res = store.search("attention mechanisms")
    assert "transformer-note" in res

    src_res = store.search_sources("transformer architecture")
    assert "s-transformer" in src_res

    # No match -> graceful message.
    assert "No results" in store.search("zzzzznonexistentquery")


# ----- index rebuild from markdown -----

def test_rebuild_index_from_markdown(store, tmp_path):
    store.capture_source("rb-src", "RB Source", "Evidence body for rebuild.", "url", url="https://x/rb")
    store.write_summary(
        "rb-note", "RB Note", "mechanism",
        "Cites it (source: rb-src) and extends (extends: rb-src).",
        tags=["rebuild", "test"],
    )
    store.retract_source("rb-src", "just testing retraction survival")

    # Nuke the entire derived index; a fresh store must rebuild from the markdown.
    index_dir = tmp_path / "wiki-library" / ".index"
    import shutil
    shutil.rmtree(index_dir)

    store2 = WikiStore(tmp_path / "wiki-library")
    # Metadata restored.
    assert "RB Note" in store2.read("rb-note")
    assert "mechanism" in store2.note_info("rb-note")
    assert "rebuild" in store2.list_tags()
    # Citation restored.
    assert "cites 1 source(s)" in store2.read("rb-note")
    # Relation restored.
    assert "extends" in store2.graph_neighbors("rb-note")
    # Retraction survived (via retractions.jsonl ground truth).
    assert "RETRACTED" in store2.read_source("rb-src")
    assert "retracted" in store2.audit()  # summary now cites a retracted source

    # Explicit rebuild is idempotent.
    out = store2.rebuild_index()
    assert "1 source(s), 1 note(s)" in out
    assert "cites 1 source(s)" in store2.read("rb-note")


# ----- id validation rejects path escapes -----

@pytest.mark.parametrize("bad_id", ["../escape", "a/b", ".hidden", "with space", "../../etc/passwd", ""])
def test_source_id_validation_rejects_escapes(store, tmp_path, bad_id):
    msg = store.capture_source(bad_id, "T", "content", "capture")
    assert "Invalid source id" in msg or "required" in msg or "empty" in msg.lower()
    # Nothing escaped the sources dir.
    wiki = tmp_path / "wiki-library"
    assert not (wiki.parent / "escape.md").exists()
    assert not (wiki.parent / "passwd").exists()


@pytest.mark.parametrize("bad_slug", ["../escape", "a/b", ".hidden", "with space"])
def test_summary_slug_validation_rejects_escapes(store, bad_slug):
    msg = store.write_summary(bad_slug, "T", "idea", "content")
    assert "Invalid slug" in msg


def test_invalid_note_type_rejected(store):
    msg = store.write_summary("t", "T", "nonsense-type", "content")
    assert "Invalid type" in msg


# ----- tool layer wiring (one tool per action) -----

_ALL_TOOLS = WIKI_READ_TOOLS + WIKI_WRITE_TOOLS


def test_read_and_write_tool_groups_are_disjoint_and_complete():
    read_names = {t.name for t in WIKI_READ_TOOLS}
    write_names = {t.name for t in WIKI_WRITE_TOOLS}
    # Disjoint: no tool is both readable and writable.
    assert read_names.isdisjoint(write_names)
    # Complete: every tool in either group has an executor, and vice versa.
    assert read_names | write_names == set(WIKI_EXECUTORS)
    # All read tools are prefixed wiki_; writers keep explicit verbs.
    assert all(n.startswith("wiki_") for n in read_names)
    assert write_names == {"wiki_capture_source", "wiki_write_summary", "wiki_retract_source"}


def test_every_tool_has_a_loadable_description_file():
    prompt_dir = Path(__file__).resolve().parents[1] / "prompt" / "tools"
    for tool in _ALL_TOOLS:
        path = prompt_dir / f"{tool.name}.md"
        assert path.exists(), f"missing description file for {tool.name}"
        assert path.read_text(encoding="utf-8").strip(), f"empty description for {tool.name}"
        # ToolSpec.description is loaded verbatim from that file.
        assert tool.description == path.read_text(encoding="utf-8").strip()


def test_doctrine_lines_preserved_in_descriptions():
    by_name = {t.name: t for t in _ALL_TOOLS}
    assert "Only sources are evidence" in by_name["wiki_read"].description
    assert "navigation, not support" in by_name["wiki_write_summary"].description
    # The wiki writers stay separate from the lab-scoped `write` tool.
    assert "write" in by_name["wiki_write_summary"].description.lower()


def test_tool_schemas_are_tight_per_action():
    for tool in _ALL_TOOLS:
        schema = tool.input_schema
        props = set(schema.get("properties", {}))
        required = set(schema.get("required", []))
        # No unused params: everything required must be a declared property.
        assert required <= props, f"{tool.name} requires undeclared props"
    # A couple of concrete checks that action-specific params did not leak across tools.
    by_name = {t.name: t for t in _ALL_TOOLS}
    assert set(by_name["wiki_read"].input_schema["properties"]) == {"slug"}
    assert set(by_name["wiki_graph_edges"].input_schema["properties"]) == {"relation"}
    assert set(by_name["wiki_tags"].input_schema["properties"]) == set()


def test_tool_executors_roundtrip(store):
    r = execute_wiki_tool(
        _call("wiki_capture_source", id="tool-src", title="T", content="body text", url="https://x/z"),
        store,
    )
    assert "saved" in r

    r = execute_wiki_tool(
        _call("wiki_write_summary", slug="tool-note", title="TN", type="idea",
              content="Note body (source: tool-src)."),
        store,
    )
    assert "saved as summary/tool-note.md" in r

    r = execute_wiki_tool(_call("wiki_read", slug="tool-note"), store)
    assert "Note body" in r
    assert "cites 1 source(s)" in r

    r = execute_wiki_tool(_call("wiki_search", query="body"), store)
    assert "tool-note" in r

    r = execute_wiki_tool(_call("wiki_graph_orphans"), store)
    assert "tool-note" in r  # idea with no relations

    # Missing required arg surfaces a helpful error, not an exception.
    assert "required" in execute_wiki_tool(_call("wiki_read"), store)
    assert "required" in execute_wiki_tool(_call("wiki_retract_source", id="tool-src"), store)

    # An unknown tool name is handled gracefully.
    assert "Unknown wiki tool" in execute_wiki_tool(_call("wiki_nope"), store)


def test_wiki_capture_without_url_is_testimony(store):
    execute_wiki_tool(
        _call("wiki_capture_source", id="cap-1", title="Cap", content="a chat message", author="alice"),
        store,
    )
    info = store.source_info("cap-1")
    assert "origin: capture" in info
    assert "author: alice" in info
