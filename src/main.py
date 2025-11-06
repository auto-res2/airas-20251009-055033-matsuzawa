# src/main.py
"""Main orchestrator – spawns ``src.train`` in a clean subprocess.

Usage (spec-compliant):
    uv run python -u -m src.main run=<run_id> results_dir=<path> mode=trial|full
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd


@hydra.main(config_path="../config", version_base=None)
def _main(cfg) -> None:
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    env = os.environ.copy()
    project_root = Path(get_original_cwd())
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run.run_id}",  # hydra group override to load correct config/run/*.yaml
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    print("[MAIN] launching: \n  ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    _main()