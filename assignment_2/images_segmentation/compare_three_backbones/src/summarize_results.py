from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    rows = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        metrics_csv = exp_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        df = pd.read_csv(metrics_csv)
        if df.empty:
            continue
        best = df.loc[df["miou"].idxmax()].to_dict()
        best["experiment"] = exp_dir.name
        rows.append(best)

    if not rows:
        raise RuntimeError("No experiment metrics found.")

    out_dir = experiments_dir.parent / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)[["experiment", "epoch", "miou", "dice", "pixel_acc", "val_loss"]]
    summary.to_csv(out_dir / "results_summary.csv", index=False)

    for metric in ["miou", "dice", "pixel_acc"]:
        plt.figure(figsize=(8, 5))
        plt.bar(summary["experiment"], summary[metric])
        plt.ylabel(metric)
        plt.title(f"Backbone comparison - {metric}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_bar.png", dpi=200)
        plt.close()

    print(f"Saved summary to {out_dir}")


if __name__ == "__main__":
    main()
