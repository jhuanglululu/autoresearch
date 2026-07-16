"""Weights on disk as safetensors, with a small JSON sidecar for provenance.

Training runs never resume — only the best (lowest val loss) weights are written
out, as an artifact of a finished run, never as state to pick a run back up from.

model.safetensors  the model weights (safetensors: safe to load, mmap-able,
                   portable). `save_model` handles the tied tok_emb/lm_head
                   weight automatically (shared-tensor bookkeeping lives in the
                   file's own str->str metadata).
model.json         sidecar: provenance metadata — the ModelConfig + step +
                   val_loss that produced these weights.

Why a sidecar JSON instead of safetensors metadata: safetensors metadata is
str->str only, so the config (ints/floats/nested) would have to be JSON-encoded
into a string there anyway. A standalone sidecar keeps it typed, human-readable,
diffable, and inspectable WITHOUT opening the weight file — and it leaves the
file's own metadata free for the shared-tensor bookkeeping `save_model` needs
for weight tying.
"""
from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_model

WEIGHTS_NAME = "model.safetensors"
SIDECAR_NAME = "model.json"


def save_checkpoint(model, step: int, val_loss: float, run_dir=".") -> None:
    """Write model.safetensors + model.json into run_dir (defaults to cwd)."""
    run_dir = Path(run_dir)
    save_model(model, str(run_dir / WEIGHTS_NAME))
    sidecar = {
        "config": vars(model.cfg),
        "step": int(step),
        "val_loss": float(val_loss),
        "vocab_size": model.cfg.vocab_size,
    }
    (run_dir / SIDECAR_NAME).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
