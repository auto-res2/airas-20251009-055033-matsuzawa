#!/usr/bin/env python
"""Main orchestrator script.

Usage examples:

(1) Smoke test – lightweight synthetic run to verify that *all* variations
    execute without GPU OOM etc.

    uv run python -m src.main --smoke-test --results-dir /tmp/zorro_results

(2) Full experiment – reads all variations from config/full_experiment.yaml

    uv run python -m src.main --full-experiment --results-dir /path/to/res
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true")
    g.add_argument("--full-experiment", action="store_true")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory to store *all* outputs (logs, figs, metrics)")
    return p.parse_args()


# ------------------------------------------------------------------- helpers #

def _tee(stream, tee_to_file):
    """Read `stream` byte-by-byte, write both to sys.<out/err> and file."""
    for line in iter(stream.readline, b""):
        decoded = line.decode()
        tee_to_file.write(decoded)
        tee_to_file.flush()
        sys.stdout.write(decoded) if tee_to_file.name.endswith("stdout.log") else sys.stderr.write(decoded)
    stream.close()


def _launch_train(run_cfg: dict, results_root: Path, python_bin: str = sys.executable):
    run_id = run_cfg["run_id"]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write single-run YAML for the sub-process to consume
    run_cfg_path = run_dir / "config.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    # Prepare log files
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_f = stdout_path.open("w")
    stderr_f = stderr_path.open("w")

    cmd = [python_bin, "-m", "src.train", "--config-path", str(run_cfg_path),
           "--results-dir", str(run_dir)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Real-time tee of stdout / stderr
    threads: List[threading.Thread] = []
    threads.append(threading.Thread(target=_tee, args=(proc.stdout, stdout_f)))
    threads.append(threading.Thread(target=_tee, args=(proc.stderr, stderr_f)))
    for t in threads:
        t.daemon = True
        t.start()

    proc.wait()
    for t in threads:
        t.join()
    stdout_f.close()
    stderr_f.close()

    if proc.returncode != 0:
        raise RuntimeError(f"Run {run_id} failed with exit code {proc.returncode}")


# -------------------------------------------------------------------- main #

def main():
    args = parse_args()
    results_root = Path(args.results_dir).expanduser()
    if results_root.exists():
        shutil.rmtree(results_root, ignore_errors=True)
    results_root.mkdir(parents=True, exist_ok=True)

    cfg_path = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    config = yaml.safe_load(cfg_path.read_text())
    experiments = config["experiments"]

    for run_cfg in experiments:
        _launch_train(run_cfg, results_root)

    # After all runs -> aggregate
    subprocess.run([sys.executable, "-m", "src.evaluate", "--results-dir", str(results_root)], check=True)


if __name__ == "__main__":
    main()