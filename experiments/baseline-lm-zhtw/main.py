"""Run entry point, invoked by the GPU worker inside the sandbox container.

Reads run_config.toml from the run directory (mounted RW), trains, and writes
metrics.json + model.pt there. No CLI args, no env vars.

TODO(implement): baseline ~50M decoder-only model + training/eval loop.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
