"""Two-tier wiki store, ported/adapted from ../research-bot/src/wiki/store.py.

Ground truth is markdown folders (shared across all goals):

    wiki-library/sources/<id>.md    immutable evidence (origin: url|capture|experiment);
                                    retract-not-delete, with reason + optional superseded_by
    wiki-library/summary/<slug>.md  editable synthesis; must cite via inline (source: id);
                                    writes citing unknown ids are REJECTED

Every markdown file is self-describing: a small ``---`` frontmatter block carries the
metadata (title, type, tags, origin, ...) and the body carries the prose with its
inline references. That makes the whole search/graph index REBUILDABLE from the
markdown alone — :meth:`WikiStore.rebuild_index` drops ``.index/`` and repopulates it.

Beside the files:

    wiki-library/.blobs/<hash>.md   content-addressed version history (append-only)
    wiki-library/timeline.jsonl     append-only change log
    wiki-library/retractions.jsonl  append-only retraction ledger (ground truth, so a
                                    retraction survives an index rebuild)
    wiki-library/.index/            REBUILDABLE index — sqlite (FTS5 + citations +
                                    relations + metadata) and, if chromadb is importable,
                                    a semantic collection. Never the source of truth.

Search runs two paths — lexical (stdlib sqlite3 FTS5/BM25) and, when chromadb is
available, semantic (embeddings) — merges them and appends a divergence warning when
the top results overlap < 50% (carried over from research-bot). With chromadb absent
the store degrades gracefully to lexical-only.

NEW versus research-bot (DESIGN.md — Knowledge base):

- typed notes:  every summary declares a ``type`` in {paper, mechanism, idea,
                experiment, result} (frontmatter).
- typed links:  inline ``(extends: slug)`` / ``(combines: a, b)`` / ``(refutes: slug)``
                references, indexed like citations, so agents can query the idea graph
                (see the graph_* methods) instead of only full-text search.

Dropped from research-bot: the tamper-evidence layer (.snapshots/, selections.jsonl)
— its own devs removed it, and there is a single trusted operator here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:  # semantic search is an optional extra (pyproject: autoresearch[index])
    import chromadb
    from chromadb.utils import embedding_functions

    _HAVE_CHROMA = True
except ImportError:  # degrade gracefully to lexical-only
    chromadb = None  # type: ignore[assignment]
    embedding_functions = None  # type: ignore[assignment]
    _HAVE_CHROMA = False

log = logging.getLogger(__name__)

# Slugs/ids become file names under the wiki dir; restrict to a safe charset
# (no slashes, no leading dot, no "..") so a model-supplied id can never escape it.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Inline citation convention in summary bodies: "(source: some-id)". Only *sources*
# are evidence, so a citation must resolve to a source id (never another summary).
_CITE_RE = re.compile(r"\(source:\s*`?([A-Za-z0-9][A-Za-z0-9._-]*)`?\s*\)")

# Inline typed-relation convention: "(extends: slug)", "(combines: a, b)",
# "(refutes: slug)". The target list is comma/whitespace separated.
_REL_RE = re.compile(r"\((extends|combines|refutes):\s*([^)]*)\)")

NOTE_TYPES = ("paper", "mechanism", "idea", "experiment", "result")
RELATION_TYPES = ("extends", "combines", "refutes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fts_match(query: str) -> str | None:
    """Turn a free-text query into an FTS5 MATCH expression (quoted tokens, OR'd)."""
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _divergence_note(semantic_ids: list[str], lexical_ids: list[str]) -> str:
    """Warn when the two retrieval paths mostly disagree (< 50% top-result overlap).

    Only meaningful when both paths ran; with lexical-only (no chromadb) there is
    nothing to diverge from, so this returns "".
    """
    if not semantic_ids or not lexical_ids:
        return ""
    overlap = len(set(semantic_ids) & set(lexical_ids))
    floor = min(len(semantic_ids), len(lexical_ids))
    if overlap / floor < 0.5:
        return (
            f"\n\n[!] Retrieval paths disagree (overlap {overlap}/{floor}): semantic and "
            "lexical search returned mostly different results. Treat these as low-confidence "
            "— read the candidates and verify claims against original sources before relying "
            "on them."
        )
    return ""


def _split_targets(raw: str) -> list[str]:
    """Split a relation target list like 'a, b c' into ['a', 'b', 'c']."""
    return [t for t in re.split(r"[,\s]+", raw.strip()) if t]


def _dump_frontmatter(meta: dict[str, str]) -> str:
    """Render a minimal ``---`` frontmatter block. Empty values are omitted."""
    lines = ["---"]
    for key, value in meta.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` frontmatter from a body. Returns ({}, text) if none present."""
    if not text.startswith("---\n") and text != "---":
        return {}, text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, text
    head, body = parts
    meta: dict[str, str] = {}
    for line in head.splitlines():
        line = line.strip()
        if not line or line == "---" or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


class WikiStore:
    """Persistent research store with two halves:

    - summaries: editable, typed markdown notes in ``summary/`` (must cite sources)
    - sources:   immutable raw markdown in ``sources/`` (the only evidence)

    Every modification is appended to an append-only ``timeline.jsonl``; every version
    ever written is content-addressed in ``.blobs/``. The search/graph index under
    ``.index/`` is derived and can always be regenerated with :meth:`rebuild_index`.
    """

    def __init__(self, wiki_dir: Path):
        self._wiki_dir = Path(wiki_dir)
        self._summary_dir = self._wiki_dir / "summary"
        self._sources_dir = self._wiki_dir / "sources"
        self._index_dir = self._wiki_dir / ".index"
        self._blobs_dir = self._wiki_dir / ".blobs"
        for d in (self._summary_dir, self._sources_dir, self._index_dir, self._blobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "wiki.db"
        self._chroma_path = self._index_dir / "chroma"
        self._timeline_path = self._wiki_dir / "timeline.jsonl"
        self._retractions_path = self._wiki_dir / "retractions.jsonl"
        self._open()
        # First run against an existing library (or a wiped index): repopulate from
        # the markdown so the store is usable without an explicit rebuild call.
        if self._index_empty() and self._has_markdown():
            self.rebuild_index()

    # ----- connection lifecycle -----

    def _open(self) -> None:
        """(Re)open the sqlite schema and the optional chroma collections."""
        self._init_db()
        self._summaries = None
        self._sources_coll = None
        if _HAVE_CHROMA:
            client = chromadb.PersistentClient(path=str(self._chroma_path))
            embedder = embedding_functions.DefaultEmbeddingFunction()
            self._summaries = client.get_or_create_collection(
                "summary", embedding_function=embedder
            )
            self._sources_coll = client.get_or_create_collection(
                "sources", embedding_function=embedder
            )

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    slug         TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    filepath     TEXT NOT NULL,
                    note_type    TEXT NOT NULL,
                    tags         TEXT NOT NULL DEFAULT '[]',
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    content_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id           TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    filepath     TEXT NOT NULL,
                    origin       TEXT NOT NULL,
                    url          TEXT,
                    author       TEXT,
                    created_at   TEXT NOT NULL,
                    content_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retractions (
                    source_id     TEXT PRIMARY KEY,
                    reason        TEXT NOT NULL,
                    superseded_by TEXT,
                    created_at    TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    slug     TEXT NOT NULL,
                    cited_id TEXT NOT NULL,
                    PRIMARY KEY (slug, cited_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    src_slug TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    dst_id   TEXT NOT NULL,
                    dst_kind TEXT NOT NULL,
                    PRIMARY KEY (src_slug, relation, dst_id)
                )
            """)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS summary_fts "
                "USING fts5(slug UNINDEXED, title, content)"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS source_fts "
                "USING fts5(id UNINDEXED, title, content)"
            )

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _index_empty(self) -> bool:
        with self._db() as conn:
            n_a = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            n_s = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        return n_a == 0 and n_s == 0

    def _has_markdown(self) -> bool:
        return any(self._summary_dir.glob("*.md")) or any(self._sources_dir.glob("*.md"))

    # ----- index rebuild (the index is always derivable from the markdown) -----

    def rebuild_index(self) -> str:
        """Wipe ``.index/`` and repopulate it from the markdown + retraction ledger.

        The markdown folders (plus retractions.jsonl) are canonical; this reconstructs
        every derived structure — metadata tables, FTS rows, citations, relations, and
        the semantic collection when chromadb is available.
        """
        if self._db_path.exists():
            self._db_path.unlink()
        if self._chroma_path.exists():
            shutil.rmtree(self._chroma_path, ignore_errors=True)
        self._open()

        retractions = self._read_retraction_ledger()
        n_sources = 0
        n_notes = 0
        with self._db() as conn:
            for path in sorted(self._sources_dir.glob("*.md")):
                meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
                sid = path.stem
                self._save_blob(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT OR REPLACE INTO sources "
                    "(id, title, filepath, origin, url, author, created_at, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        meta.get("title", sid),
                        str(path),
                        meta.get("origin", "capture"),
                        meta.get("url") or None,
                        meta.get("author") or None,
                        meta.get("captured", _now()),
                        _sha256(body),
                    ),
                )
                conn.execute(
                    "INSERT INTO source_fts (id, title, content) VALUES (?, ?, ?)",
                    (sid, meta.get("title", sid), body),
                )
                n_sources += 1
            source_ids = {r[0] for r in conn.execute("SELECT id FROM sources")}

            for source_id, ret in retractions.items():
                if source_id in source_ids:
                    conn.execute(
                        "INSERT OR REPLACE INTO retractions "
                        "(source_id, reason, superseded_by, created_at) VALUES (?, ?, ?, ?)",
                        (source_id, ret["reason"], ret.get("superseded_by"), ret["created_at"]),
                    )

            note_slugs = {p.stem for p in self._summary_dir.glob("*.md")}
            for path in sorted(self._summary_dir.glob("*.md")):
                slug = path.stem
                raw = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(raw)
                self._save_blob(raw)
                tags = [t for t in (meta.get("tags", "").split(",")) if t.strip()]
                tags = [t.strip() for t in tags]
                created = meta.get("created") or _now()
                updated = meta.get("updated") or created
                conn.execute(
                    "INSERT OR REPLACE INTO articles "
                    "(slug, title, filepath, note_type, tags, created_at, updated_at, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        slug,
                        meta.get("title", slug),
                        str(path),
                        meta.get("type", "idea"),
                        json.dumps(tags),
                        created,
                        updated,
                        _sha256(body),
                    ),
                )
                conn.execute(
                    "INSERT INTO summary_fts (slug, title, content) VALUES (?, ?, ?)",
                    (slug, meta.get("title", slug), body),
                )
                for cid in dict.fromkeys(_CITE_RE.findall(body)):
                    if cid in source_ids and cid != slug:
                        conn.execute(
                            "INSERT OR IGNORE INTO citations (slug, cited_id) VALUES (?, ?)",
                            (slug, cid),
                        )
                for relation, dst_id, dst_kind in self._extract_relations(
                    body, slug, source_ids, note_slugs
                ):
                    conn.execute(
                        "INSERT OR IGNORE INTO relations "
                        "(src_slug, relation, dst_id, dst_kind) VALUES (?, ?, ?, ?)",
                        (slug, relation, dst_id, dst_kind),
                    )
                n_notes += 1

        # Repopulate the semantic collections from the same bodies.
        if self._summaries is not None:
            with self._db() as conn:
                for row in conn.execute(
                    "SELECT slug, title, note_type, tags FROM articles"
                ).fetchall():
                    _, body = _parse_frontmatter(
                        Path(
                            conn.execute(
                                "SELECT filepath FROM articles WHERE slug = ?", (row["slug"],)
                            ).fetchone()[0]
                        ).read_text(encoding="utf-8")
                    )
                    self._summaries.upsert(
                        ids=[row["slug"]],
                        documents=[body],
                        metadatas=[{
                            "slug": row["slug"],
                            "title": row["title"],
                            "type": row["note_type"],
                            "tags": ",".join(json.loads(row["tags"])),
                        }],
                    )
                for row in conn.execute(
                    "SELECT id, title, origin, url, author FROM sources"
                ).fetchall():
                    _, body = _parse_frontmatter(
                        Path(
                            conn.execute(
                                "SELECT filepath FROM sources WHERE id = ?", (row["id"],)
                            ).fetchone()[0]
                        ).read_text(encoding="utf-8")
                    )
                    self._sources_coll.upsert(
                        ids=[row["id"]],
                        documents=[body],
                        metadatas=[{
                            "id": row["id"],
                            "title": row["title"],
                            "origin": row["origin"],
                            "url": row["url"] or "",
                            "author": row["author"] or "",
                        }],
                    )

        log.info("Wiki index rebuilt: %d source(s), %d note(s)", n_sources, n_notes)
        return f"Index rebuilt from markdown: {n_sources} source(s), {n_notes} note(s)."

    def _extract_relations(
        self, body: str, slug: str, source_ids: set[str], note_slugs: set[str]
    ) -> list[tuple[str, str, str]]:
        """Return resolvable (relation, dst_id, dst_kind) edges found inline in a body.

        Edges to unknown targets are dropped (kept clean rather than dangling); the
        write path warns the author about them.
        """
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for relation, raw in _REL_RE.findall(body):
            for dst in _split_targets(raw):
                if dst == slug or (relation, dst) in seen:
                    continue
                seen.add((relation, dst))
                if dst in source_ids:
                    out.append((relation, dst, "source"))
                elif dst in note_slugs:
                    out.append((relation, dst, "summary"))
        return out

    # ----- timeline + retraction ledger (append-only ground truth) -----

    def _log(self, action: str, kind: str, slug: str, title: str) -> None:
        entry = {"ts": _now(), "action": action, "kind": kind, "slug": slug, "title": title}
        with self._timeline_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _read_retraction_ledger(self) -> dict[str, dict]:
        """Last-write-wins map of source_id -> retraction record from the ledger."""
        out: dict[str, dict] = {}
        if not self._retractions_path.exists():
            return out
        for line in self._retractions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["source_id"]] = rec
        return out

    def history(self, n: int = 10) -> str:
        if not self._timeline_path.exists():
            return "The timeline is empty."
        lines = [ln for ln in self._timeline_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return "The timeline is empty."
        recent = lines[-n:] if n and n > 0 else lines
        out = []
        for ln in reversed(recent):
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.append(f"- {e['ts']} — {e['action']} {e['kind']} `{e['slug']}` ({e.get('title', '')})")
        if not out:
            return "The timeline is empty."
        return f"Last {len(out)} change(s):\n" + "\n".join(out)

    # ----- content-addressed blobs (append-only version history) -----

    def _save_blob(self, content: str) -> str:
        h = _sha256(content)
        path = self._blobs_dir / f"{h}.md"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return h

    # ----- summaries (editable, typed notes) -----

    def _article_path(self, slug: str) -> Path:
        return self._summary_dir / f"{slug}.md"

    def write_summary(
        self,
        slug: str,
        title: str,
        note_type: str,
        content: str,
        tags: list[str] | None = None,
    ) -> str:
        """Create or update a typed summary note.

        Citations and typed relations are declared INLINE in ``content``:
        ``(source: id)`` cites evidence, ``(extends|combines|refutes: a, b)`` links
        notes. A citation to an unknown source id — or to a summary slug (summaries are
        never evidence) — REJECTS the write.
        """
        if not _ID_RE.match(slug):
            return f"Invalid slug {slug!r} — use letters, digits, dot, underscore and hyphens (e.g. 'rope-scaling')."
        if note_type not in NOTE_TYPES:
            return f"Invalid type {note_type!r} — use one of: {', '.join(NOTE_TYPES)}."
        tags = tags or []

        with self._db() as conn:
            source_ids = {r[0] for r in conn.execute("SELECT id FROM sources")}
            note_slugs = {r[0] for r in conn.execute("SELECT slug FROM articles")}
        note_slugs.add(slug)

        cited = list(dict.fromkeys(_CITE_RE.findall(content)))
        unknown = [c for c in cited if c not in source_ids and c != slug]
        miscited_summaries = [c for c in unknown if c in note_slugs]
        truly_unknown = [c for c in unknown if c not in note_slugs]
        if miscited_summaries:
            return (
                "Not saved — these ids are summaries, not sources, so they cannot be cited "
                "as evidence: " + ", ".join(f"'{c}'" for c in miscited_summaries)
                + ". Only sources are evidence; link a related note with a typed relation "
                "instead, e.g. '(extends: " + miscited_summaries[0] + ")'."
            )
        if truly_unknown:
            return (
                "Not saved — cites unknown source id(s): "
                + ", ".join(f"'{c}'" for c in truly_unknown)
                + ". Capture the source(s) first (or check the spelling), then retry."
            )
        citation_ids = [c for c in cited if c in source_ids]
        relations = self._extract_relations(content, slug, source_ids, note_slugs)
        dropped_rel = self._dropped_relations(content, slug, source_ids, note_slugs)

        now = _now()
        with self._db() as conn:
            row = conn.execute(
                "SELECT created_at FROM articles WHERE slug = ?", (slug,)
            ).fetchone()
            created_at = row["created_at"] if row else now

        path = self._article_path(slug)
        frontmatter = _dump_frontmatter({
            "title": title,
            "type": note_type,
            "tags": ", ".join(tags),
            "created": created_at,
            "updated": now,
        })
        raw = frontmatter + "\n\n" + content.strip() + "\n"
        path.write_text(raw, encoding="utf-8")
        self._save_blob(raw)

        with self._db() as conn:
            conn.execute("""
                INSERT INTO articles
                    (slug, title, filepath, note_type, tags, created_at, updated_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    title = excluded.title, filepath = excluded.filepath,
                    note_type = excluded.note_type, tags = excluded.tags,
                    updated_at = excluded.updated_at, content_hash = excluded.content_hash
            """, (slug, title, str(path), note_type, json.dumps(tags), created_at, now, _sha256(content)))
            conn.execute("DELETE FROM summary_fts WHERE slug = ?", (slug,))
            conn.execute(
                "INSERT INTO summary_fts (slug, title, content) VALUES (?, ?, ?)",
                (slug, title, content),
            )
            conn.execute("DELETE FROM citations WHERE slug = ?", (slug,))
            for cid in dict.fromkeys(citation_ids):
                conn.execute(
                    "INSERT OR IGNORE INTO citations (slug, cited_id) VALUES (?, ?)", (slug, cid)
                )
            conn.execute("DELETE FROM relations WHERE src_slug = ?", (slug,))
            for relation, dst_id, dst_kind in relations:
                conn.execute(
                    "INSERT OR IGNORE INTO relations (src_slug, relation, dst_id, dst_kind) "
                    "VALUES (?, ?, ?, ?)",
                    (slug, relation, dst_id, dst_kind),
                )

        if self._summaries is not None:
            self._summaries.upsert(
                ids=[slug],
                documents=[content],
                metadatas=[{
                    "slug": slug, "title": title, "type": note_type, "tags": ",".join(tags),
                }],
            )

        self._log("write", "summary", slug, title)
        log.info("Wiki summary written: %s (%s)", slug, note_type)
        msg = f"Summary '{title}' saved as summary/{slug}.md (type: {note_type})."
        if not citation_ids:
            msg += (
                " [!] No source citations — the claims here are unanchored. Capture evidence "
                "with capture_source and cite it inline as '(source: id)'."
            )
        if dropped_rel:
            msg += (
                " Note: relation(s) to unknown target(s) were ignored: "
                + ", ".join(sorted(dropped_rel)) + "."
            )
        return msg

    def _dropped_relations(
        self, body: str, slug: str, source_ids: set[str], note_slugs: set[str]
    ) -> set[str]:
        dropped: set[str] = set()
        for _, raw in _REL_RE.findall(body):
            for dst in _split_targets(raw):
                if dst != slug and dst not in source_ids and dst not in note_slugs:
                    dropped.add(dst)
        return dropped

    def read(self, slug: str) -> str:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM articles WHERE slug = ?", (slug,)).fetchone()
            footer = self._evidence_footer(conn, slug) if row else ""
        if not row:
            return f"No summary found with slug '{slug}'."
        path = Path(row["filepath"])
        if not path.exists():
            return f"Summary '{slug}' is indexed but the file is missing at {path}."
        _, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        tags = json.loads(row["tags"])
        header = (
            f"**{row['title']}** (`{slug}`)\n"
            f"type: {row['note_type']} | tags: {', '.join(tags) or 'none'}\n\n"
        )
        return header + body + footer

    def _evidence_footer(self, conn: sqlite3.Connection, slug: str) -> str:
        cited = [
            r[0]
            for r in conn.execute(
                "SELECT cited_id FROM citations WHERE slug = ? ORDER BY cited_id", (slug,)
            )
        ]
        lines = ["", "", "---"]
        if cited:
            placeholders = ",".join("?" * len(cited))
            origins = {
                r["id"]: r["origin"]
                for r in conn.execute(
                    f"SELECT id, origin FROM sources WHERE id IN ({placeholders})", cited
                ).fetchall()
            }
            n_url = sum(1 for sid in cited if origins.get(sid) == "url")
            n_exp = sum(1 for sid in cited if origins.get(sid) == "experiment")
            n_other = len(cited) - n_url - n_exp
            parts = []
            if n_url:
                parts.append(f"{n_url} fetched URL(s) (externally re-verifiable)")
            if n_exp:
                parts.append(f"{n_exp} experiment run(s) (reproducible from the code snapshot)")
            if n_other:
                parts.append(f"{n_other} capture(s) (testimony — not independently verifiable)")
            lines.append(f"Evidence: cites {len(cited)} source(s) — " + ", ".join(parts) + ".")
            retracted = [sid for sid in cited if self._retraction(conn, sid)]
            if retracted:
                lines.append(
                    "[!] Cites RETRACTED source(s): "
                    + ", ".join(f"`{s}`" for s in retracted)
                    + " — re-verify the claims that relied on them."
                )
        else:
            lines.append("Evidence: no source citations — claims in this summary are unanchored.")

        # Outgoing typed relations.
        out_edges = conn.execute(
            "SELECT relation, dst_id, dst_kind FROM relations WHERE src_slug = ? "
            "ORDER BY relation, dst_id",
            (slug,),
        ).fetchall()
        if out_edges:
            by_rel: dict[str, list[str]] = {}
            for r in out_edges:
                mark = "" if r["dst_kind"] == "summary" else " (source)"
                by_rel.setdefault(r["relation"], []).append(f"`{r['dst_id']}`{mark}")
            rel_str = "; ".join(f"{rel} → {', '.join(v)}" for rel, v in by_rel.items())
            lines.append("Relations (navigation, not evidence): " + rel_str)

        # Incoming typed relations (backlinks).
        in_edges = conn.execute(
            "SELECT src_slug, relation FROM relations WHERE dst_id = ? ORDER BY relation, src_slug",
            (slug,),
        ).fetchall()
        if in_edges:
            back = ", ".join(f"`{r['src_slug']}` ({r['relation']})" for r in in_edges)
            lines.append("Referenced by: " + back + " — reuse, not verification.")
        return "\n".join(lines)

    def note_info(self, slug: str) -> str:
        with self._db() as conn:
            row = conn.execute(
                "SELECT slug, title, note_type, tags, created_at, updated_at "
                "FROM articles WHERE slug = ?",
                (slug,),
            ).fetchone()
        if not row:
            return f"No summary found with slug '{slug}'."
        tags = json.loads(row["tags"])
        return "\n".join([
            f"**{row['title']}** (`{row['slug']}`)",
            f"type: {row['note_type']}",
            f"tags: {', '.join(tags) or 'none'}",
            f"created: {row['created_at']}",
            f"updated: {row['updated_at']}",
        ])

    def list_notes(self, note_type: str | None = None, tag: str | None = None) -> str:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT slug, title, note_type, tags FROM articles ORDER BY updated_at DESC"
            ).fetchall()
        if not rows:
            return "The wiki has no summaries yet."
        lines = []
        for row in rows:
            if note_type and row["note_type"] != note_type:
                continue
            tags = json.loads(row["tags"])
            if tag and tag not in tags:
                continue
            tag_str = ", ".join(tags) if tags else "none"
            lines.append(
                f"- **{row['title']}** (`{row['slug']}`) — {row['note_type']}; tags: {tag_str}"
            )
        if not lines:
            filt = note_type or tag
            return f"No summaries matching '{filt}'."
        return "\n".join(lines)

    def list_tags(self) -> str:
        with self._db() as conn:
            rows = conn.execute("SELECT tags FROM articles").fetchall()
        counter: Counter[str] = Counter()
        for row in rows:
            counter.update(json.loads(row["tags"]))
        if not counter:
            return "No tags yet."
        lines = [
            f"- {tag} ({count})"
            for tag, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return "Tags:\n" + "\n".join(lines)

    def audit(self) -> str:
        """Cross-check every summary's evidence chain against the source layer."""
        issues = []
        with self._db() as conn:
            retracted_ids = {r[0] for r in conn.execute("SELECT source_id FROM retractions")}
            source_ids = {r[0] for r in conn.execute("SELECT id FROM sources")}
            for row in conn.execute("SELECT slug FROM articles ORDER BY slug").fetchall():
                slug = row["slug"]
                cited = [
                    r[0]
                    for r in conn.execute(
                        "SELECT cited_id FROM citations WHERE slug = ?", (slug,)
                    )
                ]
                problems = []
                bad = sorted(c for c in cited if c in retracted_ids)
                gone = sorted(c for c in cited if c not in source_ids)
                if bad:
                    problems.append("cites retracted source(s): " + ", ".join(f"`{c}`" for c in bad))
                if gone:
                    problems.append("cites missing source(s): " + ", ".join(f"`{c}`" for c in gone))
                if not cited:
                    problems.append("no source citations (unanchored)")
                if problems:
                    issues.append(f"- `{slug}`: " + "; ".join(problems))
        if not issues:
            return "Audit clean: every summary cites at least one live source."
        return (
            "Evidence audit — summaries needing attention:\n"
            + "\n".join(issues)
            + "\n\nFix by re-verifying claims, updating the summary, or capturing proper sources."
        )

    def search(self, query: str, n_results: int = 5) -> str:
        with self._db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if total == 0:
            return "The wiki has no summaries yet."

        semantic_ids: list[str] = []
        semantic_meta: dict[str, dict] = {}
        if self._summaries is not None and self._summaries.count() > 0:
            res = self._summaries.query(
                query_texts=[query], n_results=min(n_results, self._summaries.count())
            )
            semantic_ids = res["ids"][0]
            semantic_meta = dict(zip(semantic_ids, res["metadatas"][0]))

        match = _fts_match(query)
        with self._db() as conn:
            lexical_ids = []
            if match:
                lexical_ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT slug FROM summary_fts WHERE summary_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (match, n_results),
                    )
                ]
            titles = {
                r["slug"]: (r["title"], r["note_type"])
                for r in conn.execute("SELECT slug, title, note_type FROM articles").fetchall()
            }

        if not semantic_ids and not lexical_ids:
            return "No results found."
        sections = []
        if semantic_ids:
            lines = []
            for slug in semantic_ids:
                meta = semantic_meta.get(slug, {})
                lines.append(
                    f"- **{meta.get('title', slug)}** (`{slug}`) — {meta.get('type', '')}; "
                    f"tags: {meta.get('tags') or 'none'}"
                )
            sections.append("Summary search results (semantic):\n" + "\n".join(lines))
        lexical_only = [s for s in lexical_ids if s not in set(semantic_ids)]
        if lexical_only:
            label = "Lexical-only matches (BM25):" if semantic_ids else "Summary search results (lexical/BM25):"
            lines = []
            for slug in lexical_only:
                title, ntype = titles.get(slug, (slug, ""))
                lines.append(f"- **{title}** (`{slug}`) — {ntype}")
            sections.append(label + "\n" + "\n".join(lines))
        return "\n\n".join(sections) + _divergence_note(semantic_ids, lexical_ids)

    # ----- sources (immutable raw material) -----

    def _source_path(self, source_id: str) -> Path:
        return self._sources_dir / f"{source_id}.md"

    def capture_source(
        self,
        source_id: str,
        title: str,
        content: str,
        origin: str = "capture",
        url: str | None = None,
        author: str | None = None,
    ) -> str:
        """Archive raw material as an immutable source. Write-once: an existing id is
        refused (sources are never edited or deleted, only retracted)."""
        if not _ID_RE.match(source_id):
            return f"Invalid source id {source_id!r} — use letters, digits, dot, underscore and hyphens (e.g. 'arxiv-1706-03762')."
        if origin not in ("url", "capture", "experiment"):
            return f"Invalid origin {origin!r} — use 'url', 'capture' or 'experiment'."
        if not content.strip():
            return "Cannot save an empty source."
        with self._db() as conn:
            existing = conn.execute("SELECT 1 FROM sources WHERE id = ?", (source_id,)).fetchone()
        if existing:
            return f"Source '{source_id}' already exists. Sources are immutable — pick a new id."

        now = _now()
        frontmatter = _dump_frontmatter({
            "title": title,
            "origin": origin,
            "url": url or "",
            "author": author or "",
            "captured": now,
        })
        raw = frontmatter + "\n\n" + content
        path = self._source_path(source_id)
        path.write_text(raw, encoding="utf-8")
        self._save_blob(raw)

        with self._db() as conn:
            conn.execute(
                "INSERT INTO sources (id, title, filepath, origin, url, author, created_at, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, title, str(path), origin, url, author, now, _sha256(content)),
            )
            conn.execute(
                "INSERT INTO source_fts (id, title, content) VALUES (?, ?, ?)",
                (source_id, title, content),
            )
        if self._sources_coll is not None:
            self._sources_coll.upsert(
                ids=[source_id],
                documents=[content],
                metadatas=[{
                    "id": source_id, "title": title, "origin": origin,
                    "url": url or "", "author": author or "",
                }],
            )
        self._log("capture", "source", source_id, title)
        log.info("Source saved: %s (%s)", source_id, origin)
        attribution = url or author or origin
        return f"Source '{title}' saved as sources/{source_id}.md (from {attribution})."

    def retract_source(self, source_id: str, reason: str, superseded_by: str | None = None) -> str:
        """Mark a source bad/superseded. The file stays on disk (append-only); search
        excludes it and read banners it. Recorded in retractions.jsonl so it survives
        an index rebuild."""
        if not reason.strip():
            return "A retraction needs a reason."
        with self._db() as conn:
            row = conn.execute("SELECT title FROM sources WHERE id = ?", (source_id,)).fetchone()
            if not row:
                return f"No source found with id '{source_id}'."
            existing = conn.execute(
                "SELECT reason, created_at FROM retractions WHERE source_id = ?", (source_id,)
            ).fetchone()
            if existing:
                return (
                    f"Source '{source_id}' was already retracted at {existing['created_at']}: "
                    f"{existing['reason']}"
                )
            if superseded_by:
                if superseded_by == source_id:
                    return "A source cannot supersede itself."
                ok = conn.execute("SELECT 1 FROM sources WHERE id = ?", (superseded_by,)).fetchone()
                if not ok:
                    return (
                        f"Cannot mark superseded-by '{superseded_by}': no such source. "
                        "Save the replacement source first."
                    )
            created_at = _now()
            conn.execute(
                "INSERT INTO retractions (source_id, reason, superseded_by, created_at) "
                "VALUES (?, ?, ?, ?)",
                (source_id, reason, superseded_by, created_at),
            )
        with self._retractions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "source_id": source_id, "reason": reason,
                "superseded_by": superseded_by, "created_at": created_at,
            }) + "\n")
        self._log("retract", "source", source_id, row["title"])
        log.info("Source retracted: %s (%s)", source_id, reason)
        suffix = f" Superseded by '{superseded_by}'." if superseded_by else ""
        return (
            f"Source '{source_id}' retracted.{suffix} The file stays on disk (sources are "
            "append-only) but search now excludes it and read shows the retraction notice."
        )

    def _retraction(self, conn: sqlite3.Connection, source_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM retractions WHERE source_id = ?", (source_id,)
        ).fetchone()

    @staticmethod
    def _retraction_banner(ret: sqlite3.Row) -> str:
        lines = [f"[!] RETRACTED {ret['created_at']}: {ret['reason']}"]
        if ret["superseded_by"]:
            lines.append(f"Superseded by: `{ret['superseded_by']}`")
        return "\n".join(lines)

    def read_source(self, source_id: str) -> str:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
            ret = self._retraction(conn, source_id) if row else None
            cited_by = [
                r[0]
                for r in conn.execute(
                    "SELECT slug FROM citations WHERE cited_id = ? ORDER BY slug", (source_id,)
                )
            ] if row else []
        if not row:
            return f"No source found with id '{source_id}'."
        path = Path(row["filepath"])
        if not path.exists():
            return f"Source '{source_id}' is indexed but the file is missing at {path}."
        _, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        banner = self._retraction_banner(ret) + "\n\n" if ret else ""
        footer = ""
        if cited_by:
            footer = "\n\n---\nCited by: " + ", ".join(f"`{s}`" for s in cited_by) + "."
        return banner + self._source_header(row) + "\n\n" + body + footer

    def source_info(self, source_id: str) -> str:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
            ret = self._retraction(conn, source_id) if row else None
        if not row:
            return f"No source found with id '{source_id}'."
        header = self._source_header(row)
        if ret:
            header += "\n" + self._retraction_banner(ret)
        return header

    @staticmethod
    def _source_header(row: sqlite3.Row) -> str:
        lines = [f"**{row['title']}** (`{row['id']}`)", f"origin: {row['origin']}"]
        if row["url"]:
            lines.append(f"url: {row['url']}")
        if row["author"]:
            lines.append(f"author: {row['author']}")
        lines.append(f"saved: {row['created_at']}")
        return "\n".join(lines)

    def list_sources(self, author: str | None = None) -> str:
        with self._db() as conn:
            if author:
                rows = conn.execute(
                    "SELECT * FROM sources WHERE author = ? ORDER BY created_at DESC", (author,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
            retracted_ids = {r[0] for r in conn.execute("SELECT source_id FROM retractions")}
        if not rows:
            return f"No sources by '{author}'." if author else "There are no sources yet."
        lines = []
        for row in rows:
            attribution = row["url"] or row["author"] or row["origin"]
            mark = " [RETRACTED]" if row["id"] in retracted_ids else ""
            lines.append(f"- **{row['title']}** (`{row['id']}`){mark} — {attribution}")
        return "\n".join(lines)

    def search_sources(self, query: str, n_results: int = 5) -> str:
        with self._db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            retracted = {
                r["source_id"]: r for r in conn.execute("SELECT * FROM retractions").fetchall()
            }
        if total == 0:
            return "There are no sources yet."

        semantic_ids: list[str] = []
        semantic_meta: dict[str, dict] = {}
        excluded: list[str] = []
        if self._sources_coll is not None and self._sources_coll.count() > 0:
            k = min(n_results + len(retracted), self._sources_coll.count())
            res = self._sources_coll.query(query_texts=[query], n_results=k)
            raw_ids = res["ids"][0]
            semantic_meta = dict(zip(raw_ids, res["metadatas"][0]))
            excluded = [sid for sid in raw_ids if sid in retracted]
            semantic_ids = [sid for sid in raw_ids if sid not in retracted][:n_results]

        match = _fts_match(query)
        with self._db() as conn:
            lexical_ids = []
            if match:
                lexical_ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT id FROM source_fts WHERE source_fts MATCH ? "
                        "AND id NOT IN (SELECT source_id FROM retractions) "
                        "ORDER BY rank LIMIT ?",
                        (match, n_results),
                    )
                ]
            attributions = {
                r["id"]: (r["url"] or r["author"] or r["origin"], r["title"])
                for r in conn.execute("SELECT id, title, origin, url, author FROM sources").fetchall()
            }

        sections = []
        if semantic_ids:
            lines = []
            for sid in semantic_ids:
                meta = semantic_meta.get(sid, {})
                attribution = meta.get("url") or meta.get("author") or meta.get("origin", "")
                lines.append(f"- **{meta.get('title', sid)}** (`{sid}`) — {attribution}")
            sections.append("Source search results (semantic):\n" + "\n".join(lines))
        lexical_only = [sid for sid in lexical_ids if sid not in set(semantic_ids)]
        if lexical_only:
            label = "Lexical-only matches (BM25):" if semantic_ids else "Source search results (lexical/BM25):"
            lines = []
            for sid in lexical_only:
                attribution, title = attributions.get(sid, ("", sid))
                lines.append(f"- **{title}** (`{sid}`) — {attribution}")
            sections.append(label + "\n" + "\n".join(lines))
        if excluded:
            lines = []
            for sid in excluded:
                ret = retracted[sid]
                supersede = f" (superseded by `{ret['superseded_by']}`)" if ret["superseded_by"] else ""
                lines.append(f"- `{sid}` — retracted {ret['created_at']}: {ret['reason']}{supersede}")
            sections.append("Excluded retracted source(s) that matched:\n" + "\n".join(lines))
        if not sections:
            return "No matching sources found."
        return "\n\n".join(sections) + _divergence_note(semantic_ids, lexical_ids)

    # ----- graph queries over typed relations -----

    def graph_neighbors(self, slug: str, relation: str | None = None) -> str:
        """Show a note's typed edges — outgoing and incoming — optionally one relation."""
        if relation and relation not in RELATION_TYPES:
            return f"Invalid relation {relation!r} — use one of: {', '.join(RELATION_TYPES)}."
        with self._db() as conn:
            exists = conn.execute("SELECT 1 FROM articles WHERE slug = ?", (slug,)).fetchone()
            if not exists:
                return f"No summary found with slug '{slug}'."
            out_q = "SELECT relation, dst_id, dst_kind FROM relations WHERE src_slug = ?"
            in_q = "SELECT src_slug, relation FROM relations WHERE dst_id = ?"
            params_out: tuple = (slug,)
            params_in: tuple = (slug,)
            if relation:
                out_q += " AND relation = ?"
                in_q += " AND relation = ?"
                params_out = (slug, relation)
                params_in = (slug, relation)
            outgoing = conn.execute(out_q + " ORDER BY relation, dst_id", params_out).fetchall()
            incoming = conn.execute(in_q + " ORDER BY relation, src_slug", params_in).fetchall()
        lines = [f"Graph neighbours of `{slug}`" + (f" (relation: {relation})" if relation else "") + ":"]
        if outgoing:
            lines.append("Outgoing:")
            for r in outgoing:
                mark = "" if r["dst_kind"] == "summary" else " (source)"
                lines.append(f"- {r['relation']} → `{r['dst_id']}`{mark}")
        if incoming:
            lines.append("Incoming:")
            for r in incoming:
                lines.append(f"- `{r['src_slug']}` {r['relation']} → this")
        if not outgoing and not incoming:
            lines.append("(no typed relations)")
        return "\n".join(lines)

    def graph_edges(self, relation: str) -> str:
        """List every edge of a given relation type across the wiki."""
        if relation not in RELATION_TYPES:
            return f"Invalid relation {relation!r} — use one of: {', '.join(RELATION_TYPES)}."
        with self._db() as conn:
            rows = conn.execute(
                "SELECT src_slug, dst_id, dst_kind FROM relations WHERE relation = ? "
                "ORDER BY src_slug, dst_id",
                (relation,),
            ).fetchall()
        if not rows:
            return f"No '{relation}' edges yet."
        lines = [f"All '{relation}' edges:"]
        for r in rows:
            mark = "" if r["dst_kind"] == "summary" else " (source)"
            lines.append(f"- `{r['src_slug']}` {relation} `{r['dst_id']}`{mark}")
        return "\n".join(lines)

    def graph_orphans(self, note_type: str | None = None) -> str:
        """Notes with no typed relation in or out (default: only 'idea' notes — the
        orphan ideas nobody has connected to the rest of the graph)."""
        if note_type is None:
            note_type = "idea"
        if note_type not in NOTE_TYPES and note_type != "any":
            return f"Invalid type {note_type!r} — use one of: {', '.join(NOTE_TYPES)} (or 'any')."
        with self._db() as conn:
            if note_type == "any":
                rows = conn.execute("SELECT slug, note_type FROM articles ORDER BY slug").fetchall()
            else:
                rows = conn.execute(
                    "SELECT slug, note_type FROM articles WHERE note_type = ? ORDER BY slug",
                    (note_type,),
                ).fetchall()
            connected = {r[0] for r in conn.execute("SELECT src_slug FROM relations")}
            connected |= {
                r[0] for r in conn.execute("SELECT dst_id FROM relations WHERE dst_kind = 'summary'")
            }
        orphans = [r for r in rows if r["slug"] not in connected]
        if not orphans:
            label = "idea" if note_type == "idea" else note_type
            return f"No orphan {label} notes — everything is connected."
        lines = [f"Orphan {note_type} notes (no typed relations):"]
        lines += [f"- `{r['slug']}` ({r['note_type']})" for r in orphans]
        return "\n".join(lines)
