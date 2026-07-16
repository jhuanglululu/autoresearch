"""Load models.toml and goal configs.

Models are chosen ONLY here — never via CLI args or env vars. The config stores
env-var *names*; resolve_api_key() reads the actual token at call time so the
files stay safe to commit.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    base_url: str
    model: str
    api_key_env: str
    description: str = ""

    def resolve_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"env var {self.api_key_env!r} is empty — set it in .env (see .env.example)"
            )
        return key


@dataclass(frozen=True)
class ModelsConfig:
    orchestrator: ModelEndpoint
    subagent_models: tuple[ModelEndpoint, ...]

    def subagent(self, name: str) -> ModelEndpoint:
        for m in self.subagent_models:
            if m.name == name:
                return m
        raise KeyError(f"no subagent model named {name!r} in models.toml")


@dataclass(frozen=True)
class ExperimentSpec:
    """The goal's experiment domain: which baseline lab template new labs copy,
    and which read-only assets (the untouchables — e.g. a dataset + tokenizer)
    the worker resolves into every run's run_config.toml [assets] section.

    ``run_timeout_s`` is the optional per-goal host-side wall-clock kill for a run
    (``[experiment].run_timeout_s`` in the goal TOML). ``None`` means "use the
    worker's DEFAULT_TIMEOUT_S" — kept optional so existing goals stay valid."""

    baseline: Path
    assets: dict[str, Path]
    run_timeout_s: int | None = None


@dataclass(frozen=True)
class GoalConfig:
    id: str
    template_path: Path
    experiment: ExperimentSpec

    def template_text(self) -> str:
        return self.template_path.read_text(encoding="utf-8")


def load_models(path: Path | str = "models.toml") -> ModelsConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    orch = ModelEndpoint(name="orchestrator", **raw["orchestrator"])
    subs = tuple(ModelEndpoint(**entry) for entry in raw.get("subagent_model", []))
    if not subs:
        raise ValueError("models.toml defines no [[subagent_model]] entries")
    return ModelsConfig(orchestrator=orch, subagent_models=subs)


def load_goal(path: Path | str) -> GoalConfig:
    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    root = p.parent.parent  # goal files live in <root>/goals/
    template = (root / raw["template"]).resolve()
    if not template.is_file():
        raise FileNotFoundError(f"goal template not found: {template}")
    exp = raw["experiment"]
    baseline = (root / exp["baseline"]).resolve()
    if not (baseline / "pyproject.toml").is_file():
        raise FileNotFoundError(f"baseline lab template is not a uv project: {baseline}")
    # Assets may not exist yet (datasets are downloaded during setup) — resolved,
    # not validated here; the worker refuses to launch a run with a missing asset.
    assets = {name: (root / rel).resolve() for name, rel in exp.get("assets", {}).items()}
    run_timeout_s = exp.get("run_timeout_s")
    if run_timeout_s is not None:
        run_timeout_s = int(run_timeout_s)
        if run_timeout_s <= 0:
            raise ValueError("[experiment].run_timeout_s must be a positive integer (seconds)")
    return GoalConfig(
        id=raw["id"],
        template_path=template,
        experiment=ExperimentSpec(
            baseline=baseline, assets=assets, run_timeout_s=run_timeout_s
        ),
    )
