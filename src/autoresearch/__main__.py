"""Entry point:  python -m autoresearch goals/<goal>.toml

Starts the bot + orchestrator process for one goal. The single positional arg is
the goal config; models come ONLY from models.toml. The GPU worker is a separate
process: python -m autoresearch.queue.worker
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from .bot.discord_bot import DiscordChannel, SessionFeed, build_client
from .config import load_goal, load_models
from .orchestrator.loop import Orchestrator
from .queue.jobs import JobQueue
from .wiki import WikiStore

log = logging.getLogger("autoresearch")

WIKI_DIR = "wiki-library"


def _load_env(path: str | Path = ".env") -> None:
    """Tiny stdlib .env loader (no python-dotenv dependency): ``KEY=VALUE`` lines,
    ignoring blanks and ``#`` comments, never overriding an already-set env var.
    Surrounding single/double quotes on the value are stripped."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set — put it in .env (see .env.example) or the environment."
        )
    return value


def _require_int_env(name: str) -> int:
    value = _require_env(name)
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"{name} must be a numeric Discord id, got {value!r}.")


def _optional_int_env(name: str) -> int | None:
    """A numeric Discord id that may be absent (feature disabled when unset)."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"{name} must be a numeric Discord id, got {value!r}.")


async def _run(token: str, client, orchestrator: Orchestrator, ready: asyncio.Event) -> None:
    """Run the bot and orchestrator concurrently with linked shutdown: when the
    orchestrator stops we close the bot; if the bot connection ends we stop the
    orchestrator. The orchestrator waits for the channel to be bound before its first
    message so nothing is dropped at startup."""
    bot_down = asyncio.Event()

    async def run_bot() -> None:
        try:
            await client.start(token)
        except Exception:
            log.exception("Discord client stopped with an error")
        finally:
            bot_down.set()
            orchestrator.request_stop()  # bot gone -> stop the orchestrator

    async def run_orchestrator() -> None:
        ready_t = asyncio.ensure_future(ready.wait())
        down_t = asyncio.ensure_future(bot_down.wait())
        await asyncio.wait({ready_t, down_t}, return_when=asyncio.FIRST_COMPLETED)
        for t in (ready_t, down_t):
            t.cancel()
        if bot_down.is_set() and not ready.is_set():
            return  # the bot never came up; nothing to drive
        try:
            await orchestrator.run()
        finally:
            await client.close()  # orchestrator stopped -> close the bot

    await asyncio.gather(run_bot(), run_orchestrator())


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    _load_env()

    models = load_models("models.toml")
    goal = load_goal(sys.argv[1])

    token = _require_env("DISCORD_BOT_TOKEN")
    admin_id = _require_int_env("AUTORESEARCH_ADMIN")
    channel_id = _require_int_env("AUTORESEARCH_CHANNEL")
    forum_id = _optional_int_env("AUTORESEARCH_FORUM")  # optional: read-only activity feed

    log.info(
        "goal=%r orchestrator=%r subagent_models=%s forum=%s",
        goal.id, models.orchestrator.model,
        [m.name for m in models.subagent_models], forum_id,
    )

    wiki_store = WikiStore(WIKI_DIR)
    queue = JobQueue()
    dchannel = DiscordChannel()
    # The forum feed is inert until on_ready binds the channel; unset -> no factory.
    # It shares the job queue so a live run window can tail an in-flight run's log.
    feed = SessionFeed(queue=queue) if forum_id is not None else None
    orchestrator = Orchestrator(
        models, goal, channel=dchannel, wiki_store=wiki_store, queue=queue,
        session_observer_factory=feed.observer_factory if feed is not None else None,
    )

    ready = asyncio.Event()
    client = build_client(
        admin_id=admin_id,
        channel_id=channel_id,
        dchannel=dchannel,
        orchestrator=orchestrator,
        ready=ready,
        forum_id=forum_id,
        feed=feed,
    )

    asyncio.run(_run(token, client, orchestrator, ready))


if __name__ == "__main__":
    main()
