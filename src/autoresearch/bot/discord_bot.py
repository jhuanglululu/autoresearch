"""Thin Discord front-end: one channel, plain chat + slash commands.

Deliberately NOT research-bot's 1,500-line session machinery (DESIGN.md — dropped).
The bot only: relays user messages to the orchestrator, posts orchestrator
replies + periodic digests, and implements:

    /status          goal, current subagent, queue depth, token spend
    /stop            stop the orchestrator after the current subagent
    /kill-subagent   kill the running subagent; orchestrator asks you how to proceed

Admin gate: AUTORESEARCH_ADMIN env var (Discord user id). Bot token from
DISCORD_BOT_TOKEN (see .env.example).

TODO(implement): discord.py client wiring to orchestrator.loop.
"""
