"""Entry point:  python -m autoresearch goals/<goal>.toml

Starts the bot + orchestrator process for one goal. The single positional arg is
the goal config; models come ONLY from models.toml. The GPU worker is a separate
process: python -m autoresearch.queue.worker
"""
from __future__ import annotations

import sys

from .config import load_goal, load_models


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    models = load_models("models.toml")
    goal = load_goal(sys.argv[1])
    print(f"goal={goal.id!r} orchestrator={models.orchestrator.model!r} "
          f"subagent_models={[m.name for m in models.subagent_models]}")
    raise NotImplementedError("wire up bot + orchestrator (see DESIGN.md)")


if __name__ == "__main__":
    main()
