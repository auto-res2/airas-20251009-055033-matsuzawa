#!/usr/bin/env python
"""Aggregate all run results & create comparison plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True)
    return p.parse_args()


def collect(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for sub in results_dir.iterdir():
        res_file = sub / "results.json"
        if res_file.exists():
            rows.append(json.loads(res_file.read_text()))
    if not rows:
        raise RuntimeError("No results.json found – nothing to evaluate.")
    return pd.DataFrame.from_records(rows)


def plot(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    # train loss curves
    plt.figure()
    for _, row in df.iterrows():
        xs = [h["epoch"] for h in row["train_history"]]
        ys = [h["loss"] for h in row["train_history"]]
        plt.plot(xs, ys, marker="o", label=row["run_id"])
    plt.xlabel("Epoch"); plt.ylabel("CE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.pdf"); plt.close()

    # accuracy curves
    plt.figure()
    for _, row in df.iterrows():
        xs = [h["epoch"] for h in row["train_history"]]
        ys = [h["acc"] for h in row["train_history"]]
        plt.plot(xs, ys, marker="o", label=row["run_id"])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy %"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "acc_curves.pdf"); plt.close()

    # final val acc bar
    plt.figure()
    sns.barplot(x="run_id", y="val_acc", data=df)
    for i, row in df.iterrows():
        plt.text(i, row["val_acc"] + 0.2, f"{row['val_acc']:.2f}%", ha="center")
    plt.ylabel("Validation Accuracy %"); plt.tight_layout()
    plt.savefig(out_dir / "val_acc.pdf"); plt.close()


def main():
    args = parse_args()
    results_root = Path(args.results_dir).expanduser()
    df = collect(results_root)
    plot(df, results_root / "images")
    print(json.dumps(df[["run_id", "val_acc", "val_loss"]].to_dict(orient="records")))


if __name__ == "__main__":
    main()