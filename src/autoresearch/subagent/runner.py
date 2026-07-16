"""One-shot subagent loop: initial prompt in -> tool calls -> summary out.

Toolsets by type (DESIGN.md). Wiki access is per-action, split into two groups:
WIKI_READ_TOOLS (wiki_search / wiki_read / wiki_list / wiki_info / wiki_tags /
wiki_history / wiki_audit / wiki_search_sources / wiki_read_source / wiki_list_sources /
wiki_source_info / wiki_graph_neighbors / wiki_graph_edges / wiki_graph_orphans — read-only,
safe for either type) and WIKI_WRITE_TOOLS (wiki_capture_source / wiki_write_summary /
wiki_retract_source — STRICTLY wiki-scoped, through the store so structure/invariants hold).

- executor:   WIKI_READ_TOOLS + WIKI_WRITE_TOOLS, plus its lab tools: `write` (STRICTLY
              lab-scoped: resolves inside the current lab, never wiki, never past runs/
              dirs, never dataset/tokenizer), archive read (`read_file` / `list_dir`),
              `analyze_records`, `run_experiment` (the ONLY way to launch a run — no CLI,
              no env vars). DESIGN says the executor does "research -> apply", so it also
              gets `web_search` / `fetch_page`.
- researcher: web/arXiv search (`web_search`), page fetch (`fetch_page`), WIKI_READ_TOOLS +
              WIKI_WRITE_TOOLS, archive read (`read_file` / `list_dir`), `analyze_records`.
              No `write`, no `run_experiment` — it cannot edit or execute code.

The wiki write tools and the lab-scoped `write` are separate on purpose and must never be
merged: `write` cannot produce a wiki file, the wiki writers cannot produce a lab file.

Details are written to the wiki/lab; the final message returned to the
orchestrator is a short summary. Tool descriptions load verbatim from
prompt/tools/<name>.md.

A finished session is KEPT (self.messages), so the orchestrator can ask a follow-up
question with the full context intact instead of respawning an agent to rediscover it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from autoresearch.llm.base import LLMClient, Message, ToolCall, ToolSpec
from autoresearch.subagent import tools as T
from autoresearch.wiki import (
    WIKI_EXECUTORS,
    WIKI_READ_TOOLS,
    WIKI_WRITE_TOOLS,
    WikiStore,
    execute_wiki_tool,
)

SubagentType = Literal["executor", "researcher"]

DEFAULT_MAX_ROUNDS = 40


class Subagent:
    """A one-shot subagent: assemble a per-type toolset, run the loop, keep context.

    ``run(initial_prompt)`` drives the model until it returns text with no tool calls
    (that text is the summary) or ``max_rounds`` is hit (then a truncation notice plus a
    single forced no-tools summary call). ``follow_up(question)`` re-enters the same loop
    with ``self.messages`` intact.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        type_: SubagentType,
        system_prompt: str,
        *,
        lab_dir: str | Path | None = None,
        archive_dir: str | Path | None = None,
        wiki_store: WikiStore | None = None,
        pinned_assets: list[str | Path] | None = None,
        run_experiment_callable: T.RunExperimentCallable | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        if type_ not in ("executor", "researcher"):
            raise ValueError(f"unknown subagent type: {type_!r}")
        self.llm = llm_client
        self.type = type_
        self.system_prompt = system_prompt
        self.lab_dir = Path(lab_dir) if lab_dir is not None else None
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        self.wiki_store = wiki_store
        self.pinned_assets = [Path(p) for p in (pinned_assets or [])]
        self.run_experiment_callable = run_experiment_callable
        self.max_rounds = max_rounds
        self.messages: list[Message] = []

        # Read-only roots for read_file / list_dir / analyze_records: the lab and the
        # run/labs archive, whichever this agent has. The wiki is NOT here — it has its
        # own read tools through the store.
        self._read_roots = [p for p in (self.lab_dir, self.archive_dir) if p is not None]
        self.tools = self._assemble_tools()
        self._tool_names = {t.name for t in self.tools}

    # ----- toolset assembly -----

    def _assemble_tools(self) -> list[ToolSpec]:
        common_read = [T.READ_FILE_TOOL, T.LIST_DIR_TOOL, T.ANALYZE_RECORDS_TOOL]
        research = [T.WEB_SEARCH_TOOL, T.FETCH_PAGE_TOOL]
        if self.type == "executor":
            # WIKI read+write, lab write, archive read, analysis, run + research tools.
            return [
                *WIKI_READ_TOOLS,
                *WIKI_WRITE_TOOLS,
                T.WRITE_TOOL,
                *common_read,
                T.RUN_EXPERIMENT_TOOL,
                *research,
            ]
        # researcher: WIKI read+write, archive read, analysis, research tools.
        # NO write, NO run_experiment.
        return [
            *WIKI_READ_TOOLS,
            *WIKI_WRITE_TOOLS,
            *common_read,
            *research,
        ]

    # ----- tool dispatch (never crashes the loop) -----

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        try:
            return await self._dispatch(tool_call)
        except Exception as e:  # any executor failure becomes a tool-result string
            return f"Tool {tool_call.name} raised {type(e).__name__}: {e}"

    async def _dispatch(self, tool_call: ToolCall) -> str:
        name = tool_call.name
        if name not in self._tool_names:
            return f"Tool {name!r} is not available to a {self.type} subagent."

        if name in WIKI_EXECUTORS:
            if self.wiki_store is None:
                return "Wiki is unavailable in this session."
            return execute_wiki_tool(tool_call, self.wiki_store)

        if name == "write":
            if self.lab_dir is None:
                return "write is unavailable: this session has no lab directory."
            return T.exec_write(tool_call, self.lab_dir, self.pinned_assets)
        if name == "read_file":
            return T.exec_read_file(tool_call, self._read_roots)
        if name == "list_dir":
            return T.exec_list_dir(tool_call, self._read_roots)
        if name == "analyze_records":
            if not self._read_roots:
                return "analyze_records is unavailable: no lab/archive directory."
            return T.exec_analyze_records(tool_call, self._read_roots)
        if name == "web_search":
            return await T.exec_web_search(tool_call)
        if name == "fetch_page":
            return await T.exec_fetch_page(tool_call)
        if name == "run_experiment":
            return await T.exec_run_experiment(tool_call, self.run_experiment_callable)

        return f"Unknown tool: {name}"

    # ----- the loop -----

    async def run(self, initial_prompt: str) -> str:
        """Run the assignment to a final summary string."""
        self.messages.append(Message(role="user", content=initial_prompt))
        return await self._loop()

    async def follow_up(self, question: str) -> str:
        """Ask a follow-up on a finished session, reusing the full context."""
        self.messages.append(Message(role="user", content=question))
        return await self._loop()

    async def _loop(self) -> str:
        for _ in range(self.max_rounds):
            response = await self.llm.complete(self.system_prompt, self.messages, self.tools)
            self.messages.append(response.message)
            if not response.message.tool_calls:
                return response.message.content
            for tool_call in response.message.tool_calls:
                result = await self._execute_tool(tool_call)
                self.messages.append(
                    Message(role="tool", content=result, tool_call_id=tool_call.id)
                )
        return await self._forced_summary()

    async def _forced_summary(self) -> str:
        """max_rounds reached without a summary: force one final no-tools call."""
        notice = (
            f"[reached max_rounds={self.max_rounds} without a final summary]"
        )
        self.messages.append(
            Message(
                role="user",
                content=(
                    notice + " Stop here and reply with your final summary for the "
                    "orchestrator. Do not call any tools."
                ),
            )
        )
        response = await self.llm.complete(self.system_prompt, self.messages, [])
        self.messages.append(response.message)
        return f"{notice}\n{response.message.content}"
