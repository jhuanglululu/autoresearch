"""Thin Discord front-end: one channel, plain chat + slash commands.

Deliberately NOT research-bot's 1,500-line session machinery (DESIGN.md — dropped).
The bot only relays the admin's messages to the orchestrator, posts orchestrator
replies + periodic digests back, and exposes four slash commands:

    /status          orchestrator status snapshot (no LLM turn)
    /sessions        list subagent sessions + their states
    /stop            stop the orchestrator after the current subagent
    /kill-subagent   kill a running subagent now (asks you how to proceed next)

Everything is admin-gated (AUTORESEARCH_ADMIN, a Discord user id) and confined to
one channel (AUTORESEARCH_CHANNEL, a channel id). Non-admin messages, messages in
any other channel, and the bot's own messages are ignored. Bot token comes from
DISCORD_BOT_TOKEN (see .env.example).

Mention gating (steering channel): the orchestrator doesn't know it lives in a busy
Discord channel, so to keep ambient chat out of its context an admin message must ALSO
explicitly @mention the bot to be relayed. ``on_message`` requires the mention after the
admin/channel gate, strips the mention token(s) so the orchestrator sees clean text
(``extract_command_text``), and SILENTLY ignores admin messages that don't mention the
bot. Slash commands are unaffected — they already target the bot.

Read-only forum feed (AUTORESEARCH_FORUM, optional): when set, each spawned subagent
gets its own thread in that forum channel with a live, batched trace of its activity
(``SessionFeed`` / ``SessionThread``). It is pure observability — no steering happens
there — and the whole feature is inert when the id is unset or the channel can't be
resolved. Discord errors in the feed are logged and dropped, never propagated.

The orchestrator only ever sees the ``Channel`` protocol (orchestrator/channel.py);
``DiscordChannel`` implements it with a pair of asyncio queues. ``discord`` itself is
imported lazily inside ``build_client`` so the module — and its pure helpers
(``split_message``, ``is_authorized``, ``extract_command_text``, ``DiscordChannel``,
``SessionFeed``) — import without a token or a network connection.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

DISCORD_LIMIT = 2000
_FENCE = "```"


# ----- message splitting (port + extend research-bot/src/textsplit.py) -----

def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Split ``text`` into chunks that each fit Discord's per-message limit.

    Ported from research-bot's ``textsplit`` (prefer newline boundaries; hard-split
    any single line longer than ``limit`` — e.g. a long URL) and extended so a split
    landing inside a ``` code fence stays valid markdown: the open fence is closed at
    the end of the emitted chunk and reopened (same info string) at the start of the
    next, so every chunk has balanced fences.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    fence: str | None = None  # the open fence's line (e.g. "```py"), else None
    current: str = ""  # chunk under construction; never carries its own close fence

    def reserve() -> int:
        # room kept free at the end of a chunk to close an open fence
        return len(_FENCE) + 1 if fence is not None else 0

    def fresh_max() -> int:
        # longest a single line may be to fit on a brand-new chunk
        prefix = (len(fence) + 1) if fence is not None else 0
        return max(1, limit - prefix - reserve())

    def add(line: str) -> None:
        nonlocal current
        current = line if not current else f"{current}\n{line}"

    def emit() -> None:
        nonlocal current
        piece = current
        if fence is not None and piece:
            piece = f"{piece}\n{_FENCE}"
        if piece:
            chunks.append(piece)
        current = fence if fence is not None else ""

    for line in text.split("\n"):
        # Hard-split a line too long to fit even on a fresh chunk.
        while len(line) > fresh_max():
            if current and current != fence:
                emit()  # flush real content so the long line starts fresh
                continue
            take = fresh_max()
            add(line[:take])
            emit()
            line = line[take:]

        sep = 1 if current else 0
        if len(current) + sep + len(line) + reserve() > limit:
            emit()
        add(line)

        # Toggle fence state AFTER the fence line is placed in the chunk.
        if line.lstrip().startswith(_FENCE):
            fence = line if fence is None else None

    if current and current != fence:
        tail = current
        if fence is not None:
            tail = f"{tail}\n{_FENCE}"
        chunks.append(tail)
    return chunks


# ----- admin/channel gating (pure + testable) -----

def is_authorized(
    author_id: int,
    channel_id: int | None,
    *,
    admin_id: int | None,
    bound_channel_id: int | None,
    is_bot: bool = False,
) -> bool:
    """True iff a message/command should be acted on: it must come from the admin, in
    the bound channel, and not from a bot. AUTORESEARCH_ADMIN gates EVERYTHING — if it
    is unset (``admin_id is None``) nothing is ever authorized."""
    if is_bot:
        return False
    if admin_id is None or author_id != admin_id:
        return False
    if bound_channel_id is not None and channel_id != bound_channel_id:
        return False
    return True


def extract_command_text(content: str, bot_id: int) -> str | None:
    """Return the message text with the bot's @mention token(s) removed, or ``None`` if
    the bot was not mentioned at all.

    Both discord.py mention forms are recognised — ``<@id>`` and the nickname form
    ``<@!id>`` — anywhere in the message (not only at the start). ``None`` means "this
    message does not address the bot" (the caller ignores it silently); ``""`` means the
    bot was mentioned but nothing else was said. Whitespace freed by removing the token(s)
    is collapsed so the orchestrator sees clean text.
    """
    if not content:
        return None
    pattern = re.compile(rf"<@!?{int(bot_id)}>")
    if not pattern.search(content):
        return None
    stripped = pattern.sub(" ", content)
    return " ".join(stripped.split())


# ----- the Channel implementation the orchestrator talks to -----

class DiscordChannel:
    """Implements orchestrator ``Channel`` over Discord: inbound admin messages arrive
    via ``post_inbound`` (from ``on_message``) and are read by ``recv``; ``send`` posts
    to the bound channel, chunked to Discord's limit. A Discord hiccup in ``send`` is
    logged and dropped — it must never propagate and kill the orchestrator loop.

    ``channel`` (a discord messageable) is bound once the client is ready; until then,
    and if it later goes away, sends are logged and dropped.
    """

    def __init__(
        self,
        channel: Any | None = None,
        *,
        split: Callable[[str], list[str]] = split_message,
        logger: logging.Logger | None = None,
    ) -> None:
        import asyncio

        self.channel = channel
        self._inbound: asyncio.Queue[str] = asyncio.Queue()
        self._split = split
        self._log = logger or log

    def post_inbound(self, text: str) -> None:
        """Deliver an inbound admin message to the waiting ``recv`` (non-blocking)."""
        self._inbound.put_nowait(text)

    async def recv(self) -> str:
        return await self._inbound.get()

    async def send(self, text: str) -> None:
        if not text or not text.strip():
            return
        target = self.channel
        if target is None:
            self._log.warning("no bound Discord channel yet; dropping message")
            return
        for chunk in self._split(text):
            if not chunk:
                continue
            try:
                await target.send(chunk)
            except Exception as e:  # a Discord hiccup must never kill the orchestrator
                self._log.warning(
                    "dropping Discord message chunk (%s): %s", type(e).__name__, e
                )


# ----- read-only forum feed of subagent activity (AUTORESEARCH_FORUM) -----
#
# Each session owns ONE forum thread with two DISTINCT presentations:
#   - the tool-call FEED: every ~7s flush POSTS a new message with the batch of activity
#     lines accumulated since the last flush (empty flushes skipped). Separate posts carry
#     Discord's own sent-at timestamps, so the timeline is chronological for free; session
#     end posts a final status-line message.
#   - the live RUN WINDOW: while an executor's run_experiment is in flight, ONE message is
#     EDITED in place with the tail of that run's log.txt (a ```text fence + a
#     `⏱ … · running` footer, final status on completion). It is a live status display,
#     not a log, so editing — not posting — is correct there.

FEED_FLUSH_INTERVAL_S = 7.0  # post a feed batch / refresh the run window at most ~once / 7s
_THREAD_NAME_PROMPT = 60  # chars of the prompt shown in the thread name
_PROMPT_PREVIEW = 200  # chars of the prompt shown in the thread's opening post
_SUMMARY_BRIEF = 200  # chars of the summary shown on the final status line
_STATUS_ICON = {"done": "✅", "failed": "❌", "timeout": "⏰", "killed": "⏹"}

RUN_TAIL_LINES = 20  # last N lines of log.txt shown in the live run window
RUN_TAIL_BYTES = 8192  # only the last few KB of log.txt are ever read (never the whole)
_MONSTER_LINE = 1900  # a single log line longer than this is hard-truncated
_WAITING = "(waiting for run to start …)"
_FENCE_LANG = "text"


def _format_event(event: dict[str, Any]) -> str:
    """Render one runner event (see ``subagent.runner``) as a compact single line.
    ``start`` is intentionally dropped — the thread's opening post already carries it."""
    kind = event.get("kind")
    if kind == "tool_call":
        brief = event.get("brief", "")
        return f"🔧 {event.get('tool', '')} {brief}".rstrip()
    if kind == "tool_result":
        return f"→ {' '.join((event.get('text') or '').split())}"
    if kind == "follow_up":
        return f"❓ follow-up: {' '.join((event.get('question') or '').split())}"
    if kind == "summary":
        return f"✅ done: {' '.join((event.get('text') or '').split())}"
    return ""


def _fmt_elapsed(seconds: float) -> str:
    """Compact h/m/s, e.g. ``45s`` / ``4m32s`` / ``1h04m32s``."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _render_message(lines: list[str], footer: str) -> str:
    """A ```text code block wrapping ``lines`` with ``footer`` on its own line below."""
    body = "\n".join(lines)
    return f"```{_FENCE_LANG}\n{body}\n```\n{footer}"


def _read_tail_bytes(path: Path, max_bytes: int) -> bytes:
    """Read only the last ``max_bytes`` of a file (seek from the end) — never the whole."""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read()


def _tail_log(run_dir: Path) -> str | None:
    """The last ``RUN_TAIL_LINES`` lines of ``run_dir/log.txt`` (reading only the tail
    bytes), or ``None`` if the log does not exist yet / can't be read."""
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return None
    try:
        data = _read_tail_bytes(log_path, RUN_TAIL_BYTES)
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-RUN_TAIL_LINES:])


def _fit_window(lines: list[str], footer: str) -> str:
    """Render a live run window that always fits 2000 chars: hard-truncate any monster
    line, then drop the OLDEST lines until it fits (a window, not an archive)."""
    trimmed = [
        (ln[:_MONSTER_LINE] + "…") if len(ln) > _MONSTER_LINE else ln for ln in lines
    ]
    while trimmed:
        content = _render_message(trimmed, footer)
        if len(content) <= DISCORD_LIMIT:
            return content
        trimmed.pop(0)  # drop the oldest line and retry
    return _render_message(["(log too large to preview)"], footer)


def _is_gone(exc: Exception) -> bool:
    """Whether a Discord error means the message was deleted (a 404 / NotFound) so we
    should re-post a fresh one rather than keep editing a ghost."""
    return getattr(exc, "status", None) == 404 or "NotFound" in type(exc).__name__


class SessionThread:
    """One subagent session's forum thread: a posted tool-call feed + a live run window.

    The tool-call FEED buffers activity lines and, every flush, POSTS the batch as a new
    message (split to the limit; empty flushes skipped) — separate posts keep the timeline
    chronological via Discord's own timestamps. Session end posts a final status-line
    message. While a run_experiment is in flight the RUN WINDOW is ONE message EDITED in
    place with the live tail of that run's log.txt. The thread is created lazily; every
    Discord call is swallowed (a vanished run-window message is re-posted, any other error
    logged and dropped) so nothing reaches the loop.
    """

    def __init__(
        self,
        forum: Any,
        session_id: str,
        type_: str,
        model: str,
        prompt: str,
        *,
        lab_id: str | None = None,
        queue: Any | None = None,
        interval_s: float = FEED_FLUSH_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self.forum = forum
        self.session_id = session_id
        self.type = type_
        self.model = model
        self.prompt = prompt
        self.lab_id = lab_id
        self._queue = queue
        self._interval_s = interval_s
        self._clock = clock
        self._log = logger or log
        self._started_at = clock()
        self._thread: Any | None = None
        self._closed = False
        self._task: Any | None = None

        # tool-call feed state: lines accumulated since the last posted batch
        self._feed_buffer: list[str] = []

        # live run-window state (activated by a run_experiment tool call)
        self._run_active = False
        self._run_final = False
        self._run_started = 0.0
        self._run_job_id: str | None = None
        self._run_dir: Path | None = None
        self._run_msg: Any | None = None

    def begin(self) -> None:
        """Start the periodic flusher (best-effort: no running loop -> caller drives flush)."""
        import asyncio

        try:
            self._task = asyncio.ensure_future(self._run())
        except RuntimeError:  # no running loop (e.g. a unit test) — manual flush only
            self._task = None

    async def on_event(self, event: dict[str, Any]) -> None:
        """The runner's fire-and-forget hook: append a feed line, arm/observe a run, or
        close on the end signal."""
        kind = event.get("kind")
        if kind == "end":
            await self.close(event)
            return
        line = _format_event(event)
        if line:
            self._feed_buffer.append(line)
        # A run_experiment tool call means a training run is about to start for this lab.
        if kind == "tool_call" and event.get("tool") == "run_experiment":
            self._run_begin()

    def _run_begin(self) -> None:
        if self.lab_id is None or self._queue is None:
            return  # nothing to tail without a lab + queue
        self._run_active = True
        self._run_final = False
        self._run_started = self._clock()
        self._run_job_id = None
        self._run_dir = None

    async def _run(self) -> None:  # pragma: no cover - timing loop, driven by hand in tests
        import asyncio

        while not self._closed:
            await asyncio.sleep(self._interval_s)
            await self.flush()

    # ----- lazy thread + single-message edit primitive -----

    async def _ensure_thread(self) -> None:
        if self._thread is not None or self.forum is None:
            return
        name = f"{self.session_id} · {self.prompt[:_THREAD_NAME_PROMPT]}".replace("\n", " ")
        header = (
            f"🚀 {self.session_id} · {self.type} on {self.model}\n"
            f"{self.prompt[:_PROMPT_PREVIEW]}"
        )
        try:
            created = await self.forum.create_thread(name=name[:100], content=header)
            self._thread = getattr(created, "thread", created)
        except Exception as e:  # a Discord hiccup must never reach the orchestrator
            self._log.warning(
                "forum thread create failed for %s (%s): %s",
                self.session_id, type(e).__name__, e,
            )

    async def _put(self, msg: Any | None, content: str) -> Any | None:
        """Create-or-edit one live message; return the (possibly new) message object. A
        deleted message is re-posted; any other Discord error is logged and dropped."""
        if self._thread is None:
            return msg
        if msg is None:
            try:
                return await self._thread.send(content)
            except Exception as e:
                self._log.warning(
                    "forum send failed for %s (%s): %s", self.session_id, type(e).__name__, e
                )
                return None
        try:
            await msg.edit(content=content)
            return msg
        except Exception as e:
            if _is_gone(e):  # the live message was deleted — re-post a fresh one
                try:
                    return await self._thread.send(content)
                except Exception as e2:
                    self._log.warning(
                        "forum re-post failed for %s (%s): %s",
                        self.session_id, type(e2).__name__, e2,
                    )
                    return None
            self._log.warning(
                "forum edit failed for %s (%s): %s", self.session_id, type(e).__name__, e
            )
            return msg

    # ----- elapsed footers -----

    def _elapsed_str(self) -> str:
        return _fmt_elapsed(self._clock() - self._started_at)

    def _run_elapsed_str(self) -> str:
        return _fmt_elapsed(self._clock() - self._run_started)

    # ----- publish: tool-call feed (post a batch per flush; empty flushes skipped) -----

    async def _publish_feed(self) -> None:
        if not self._feed_buffer:
            return  # skip empty flushes
        await self._ensure_thread()
        if self._thread is None:  # couldn't create the thread — drop, stay bounded
            self._feed_buffer.clear()
            return
        lines, self._feed_buffer = self._feed_buffer, []
        # Fenced so tool-call batches read as machine activity, distinct from any
        # prose in the thread. split_message is fence-aware, so oversized batches
        # keep balanced fences per chunk.
        batch = "```text\n" + "\n".join(lines) + "\n```"
        for chunk in split_message(batch):
            if not chunk:
                continue
            try:
                await self._thread.send(chunk)
            except Exception as e:  # a Discord hiccup must never reach the orchestrator
                self._log.warning(
                    "dropping forum feed batch for %s (%s): %s",
                    self.session_id, type(e).__name__, e,
                )

    async def _post_final_status(self, status: str, summary: str) -> None:
        """Post the session's closing status as its own message, e.g.
        ``✅ done · 12m08s: <summary…>``."""
        await self._ensure_thread()
        if self._thread is None:
            return
        icon = _STATUS_ICON.get(status, "ℹ️")
        line = f"{icon} {status} · {self._elapsed_str()}"
        snippet = " ".join((summary or "").split())[:_SUMMARY_BRIEF]
        if snippet:
            line += f": {snippet}"
        for chunk in split_message(line):
            if not chunk:
                continue
            try:
                await self._thread.send(chunk)
            except Exception as e:
                self._log.warning(
                    "dropping forum status line for %s (%s): %s",
                    self.session_id, type(e).__name__, e,
                )

    # ----- publish: live run window (tail of log.txt while the run is in flight) -----

    async def _publish_run(self) -> None:
        if not self._run_active:
            return
        await self._ensure_thread()
        if self._thread is None:
            return
        job = self._queue.find_running(self.lab_id) if self._queue is not None else None
        if job is not None:
            self._run_job_id = job.id
            if job.run_dir and self._run_dir is None:
                self._run_dir = Path(job.run_dir)
            gone = False
        else:
            gone = self._run_job_id is not None  # was running, now left the running/ lane

        tail = _tail_log(self._run_dir) if self._run_dir is not None else None
        body = tail.splitlines() if tail and tail.strip() else [_WAITING]

        if gone:
            await self._finalize_run(body)
            return
        footer = f"⏱ {self._run_elapsed_str()} · running"
        self._run_msg = await self._put(self._run_msg, _fit_window(body, footer))

    async def _finalize_run(self, body: list[str], status: str | None = None) -> None:
        if status is None:
            done = self._queue.get_done(self._run_job_id) if self._run_job_id else None
            status = (done.status if done and done.status else None) or "done"
        footer = f"⏱ {self._run_elapsed_str()} · {status}"
        self._run_msg = await self._put(self._run_msg, _fit_window(body, footer))
        self._run_active = False
        self._run_final = True

    # ----- flush + close -----

    async def flush(self) -> None:
        """One cadence tick: post any buffered feed lines, then refresh the run window."""
        await self._publish_feed()
        await self._publish_run()

    async def close(self, event: dict[str, Any] | None = None) -> None:
        """Settle the run window, post the last feed batch, then a final status-line
        message. Idempotent; stops the flusher."""
        if self._closed:
            return
        self._closed = True
        status = (event or {}).get("status") or "ended"
        await self._publish_feed()  # flush any remaining buffered activity lines first
        if self._run_active:
            await self._publish_run()  # a completed run detects 'gone' and finalizes here
            if self._run_active:  # still in flight at session end — stamp it terminally
                tail = _tail_log(self._run_dir) if self._run_dir is not None else None
                body = tail.splitlines() if tail and tail.strip() else [_WAITING]
                await self._finalize_run(body, status=status)
        await self._post_final_status(status, (event or {}).get("summary") or "")
        if self._task is not None:
            self._task.cancel()
            self._task = None


class SessionFeed:
    """The read-only forum feed. Owns one ``SessionThread`` per subagent session.

    ``bind`` attaches the resolved forum channel once the client is ready; until then (or
    if the id is unset / can't be resolved) ``forum`` is ``None`` and the feature is inert
    — ``observer_factory`` returns ``None`` so the orchestrator runs exactly as today. The
    ``queue`` lets a live run window locate a run's log dir while it is in flight.
    """

    def __init__(
        self,
        forum: Any | None = None,
        *,
        queue: Any | None = None,
        interval_s: float = FEED_FLUSH_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self.forum = forum
        self._queue = queue
        self._interval_s = interval_s
        self._clock = clock
        self._log = logger or log
        self.sessions: dict[str, SessionThread] = {}

    def bind(self, forum: Any | None) -> None:
        self.forum = forum

    def observer_factory(
        self,
        session_id: str,
        type_: str,
        model: str,
        prompt: str,
        lab_id: str | None = None,
    ) -> "Callable[[dict[str, Any]], Awaitable[None]] | None":
        """Return an ``on_event`` callback for a new session, or ``None`` when the feed is
        inert (no forum channel bound). Injected into the orchestrator as
        ``session_observer_factory``."""
        if self.forum is None:
            return None
        thread = SessionThread(
            self.forum, session_id, type_, model, prompt,
            lab_id=lab_id, queue=self._queue,
            interval_s=self._interval_s, clock=self._clock, logger=self._log,
        )
        self.sessions[session_id] = thread
        thread.begin()
        return thread.on_event


# ----- /btw side-channel handler (pure of Discord; testable) -----

async def answer_aside(orchestrator: Any, question: str) -> tuple[str, bool]:
    """Compute the ``/btw`` answer text and whether it is an error.

    Returns ``(text, is_error)`` and NEVER raises: the orchestrator's ``aside`` already
    turns a spent budget into a normal answer string, and any other failure here becomes
    a friendly message flagged ``is_error=True`` (the command posts those ephemerally).
    """
    try:
        answer = await orchestrator.aside(question)
    except Exception as e:  # a /btw must never crash the bot
        log.warning("/btw aside failed (%s): %s", type(e).__name__, e)
        return (f"Sorry — I couldn't answer that ({type(e).__name__}).", True)
    return (answer if answer and answer.strip() else "(no answer)", False)


# ----- the discord.py client (imports discord lazily) -----

def build_client(
    *,
    admin_id: int,
    channel_id: int,
    dchannel: DiscordChannel,
    orchestrator: Any,
    ready: Any,
    forum_id: int | None = None,
    feed: "SessionFeed | None" = None,
) -> Any:
    """Build the discord.py ``Client`` wiring the bound channel + slash commands to the
    orchestrator. ``ready`` is an ``asyncio.Event`` set once the client is connected and
    the channel is bound, so the caller can hold the orchestrator's first message until
    the channel exists (see ``__main__``). Everything is admin+channel gated.

    If ``forum_id`` and ``feed`` are given, the forum channel is resolved on ready and
    bound to the ``feed`` (the read-only subagent activity feed); when unset or not
    resolvable the feed stays inert."""
    import discord
    from discord import app_commands

    intents = discord.Intents.default()
    intents.message_content = True  # required to read message text (enable in portal)
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    @client.event
    async def on_ready() -> None:  # pragma: no cover - needs a live connection
        dchannel.channel = client.get_channel(channel_id)
        if dchannel.channel is None:
            log.warning("AUTORESEARCH_CHANNEL %s not visible to the bot", channel_id)
        if feed is not None and forum_id is not None:
            forum = client.get_channel(forum_id)
            if forum is None:
                log.warning("AUTORESEARCH_FORUM %s not visible to the bot", forum_id)
            feed.bind(forum)  # None keeps the feed inert
        try:
            await tree.sync()  # guild-agnostic: global sync
        except Exception as e:
            log.warning("slash-command sync failed: %s", e)
        log.info("logged in as %s; bound channel=%s", client.user, channel_id)
        ready.set()

    @client.event
    async def on_message(message: "discord.Message") -> None:  # pragma: no cover
        if not is_authorized(
            message.author.id,
            message.channel.id,
            admin_id=admin_id,
            bound_channel_id=channel_id,
            is_bot=bool(getattr(message.author, "bot", False)),
        ):
            return
        # Mention gate: only an admin message that @mentions the bot enters the
        # orchestrator's context; the token(s) are stripped so it sees clean text.
        content = (message.content or "").strip()
        bot_user = client.user
        text = (
            extract_command_text(content, bot_user.id) if bot_user is not None else None
        )
        if text is None:
            # Fall back to discord's parsed mention list, then silently ignore.
            if bot_user is not None and bot_user in getattr(message, "mentions", []):
                text = content
            else:
                return
        if text:
            dchannel.post_inbound(text)

    async def _guard(interaction: "discord.Interaction") -> bool:  # pragma: no cover
        if is_authorized(
            interaction.user.id,
            interaction.channel_id,
            admin_id=admin_id,
            bound_channel_id=channel_id,
        ):
            return True
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return False

    async def _reply(interaction: "discord.Interaction", text: str) -> None:  # pragma: no cover
        chunks = split_message(text) if text and text.strip() else ["(nothing to show)"]
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

    @tree.command(name="status", description="Orchestrator status snapshot (no LLM turn).")
    async def status_cmd(interaction: "discord.Interaction") -> None:  # pragma: no cover
        if not await _guard(interaction):
            return
        await _reply(interaction, orchestrator.status_text())

    @tree.command(name="sessions", description="List subagent sessions and their states.")
    async def sessions_cmd(interaction: "discord.Interaction") -> None:  # pragma: no cover
        if not await _guard(interaction):
            return
        await _reply(interaction, orchestrator.sessions_text())

    @tree.command(name="stop", description="Stop the orchestrator after the current subagent.")
    async def stop_cmd(interaction: "discord.Interaction") -> None:  # pragma: no cover
        if not await _guard(interaction):
            return
        orchestrator.request_stop()
        await interaction.response.send_message(
            "Stopping after the in-flight subagent finishes.", ephemeral=True
        )

    @tree.command(name="kill-subagent", description="Kill a running subagent now.")
    @app_commands.describe(session_id="Session id to kill (e.g. exec-1, res-2).")
    async def kill_cmd(
        interaction: "discord.Interaction", session_id: str
    ) -> None:  # pragma: no cover
        if not await _guard(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        result = await orchestrator.kill_session(session_id.strip())
        await interaction.followup.send(result, ephemeral=True)

    @tree.command(
        name="btw",
        description="Ask a side question using the orchestrator's context (no tools, no memory).",
    )
    @app_commands.describe(question="Your side question for the orchestrator.")
    async def btw_cmd(
        interaction: "discord.Interaction", question: str
    ) -> None:  # pragma: no cover
        if not await _guard(interaction):
            return
        # Non-ephemeral: the answer may be useful to see in-channel, and channel messages
        # never re-enter the orchestrator's context (mention gating + the bot filter).
        await interaction.response.defer()
        text, is_error = await answer_aside(orchestrator, question)
        chunks = split_message(text) if text.strip() else ["(no answer)"]
        await interaction.followup.send(chunks[0], ephemeral=is_error)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=is_error)

    client.tree = tree  # keep a handle (parity with research-bot; handy for tests/tools)
    return client
