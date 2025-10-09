#!/usr/bin/env python
"""Aggregate results across run variations, compute comparison metrics &
publication-ready figures.  This script is triggered *once* by src.main after
all individual experiment runs are finished."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

FIG_TOPICS = [
    ("training_loss", "Cross-entropy loss"),
    ("accuracy", "Accuracy (%)"),
    ("final_accuracy_bar", "Final validation accuracy")
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True,
                   help="Root directory containing sub-dirs for each run")
    return p.parse_args()


def collect_results(results_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    for sub in results_dir.iterdir():
        if not sub.is_dir():
            continue
        res_file = sub / "results.json"
        if res_file.exists():
            rec = json.loads(res_file.read_text())
            records.append(rec)
    if not records:
        raise RuntimeError(f"No results.json files found in {results_dir}")
    return pd.DataFrame.from_records(records)


def plot_comparisons(df: pd.DataFrame, results_dir: Path):
    images = results_dir / "images"
    images.mkdir(exist_ok=True, parents=True)

    # 1) Line curves – loss & acc
    for metric_key, ylabel in [("loss", "Cross-entropy loss"),
                               ("acc", "Accuracy (%)")]:
        plt.figure()
        for _, row in df.iterrows():
            y = [h[metric_key] for h in row["train_history"]]
            x = [h["epoch"] for h in row["train_history"]]
            plt.plot(x, y, marker="o", label=row["run_id"])
            plt.annotate(f"{y[-1]:.2f}" if metric_key == "acc" else f"{y[-1]:.3f}",
                         (x[-1], y[-1]), textcoords="offset points",
                         xytext=(0, 5), ha='center')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(ylabel + " comparison")
        plt.tight_layout()
        fname = "training_loss.pdf" if metric_key == "loss" else "accuracy.pdf"
        plt.savefig(images / fname, bbox_inches="tight")
        plt.close()

    # 2) Bar – final validation accuracy
    plt.figure()
    sns.barplot(x="run_id", y="val_acc", data=df)
    for idx, row in df.iterrows():
        plt.text(idx, row["val_acc"] + 0.2, f"{row['val_acc']:.2f}%", ha='center')
    plt.ylabel("Validation accuracy (%)")
    plt.xlabel("Run")
    plt.title("Final validation accuracy across runs")
    plt.tight_layout()
    plt.savefig(images / "final_accuracy_bar.pdf", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser()

    df = collect_results(results_dir)
    plot_comparisons(df, results_dir)

    # Output comparison results in JSON (stdout)
    cols_to_export = ["run_id", "val_acc"]
    if "val_loss" in df.columns:
        cols_to_export.append("val_loss")
    comp = df[cols_to_export].to_dict(orient="records")
    print(json.dumps({"comparison": comp}))


if __name__ == "__main__":
    main()