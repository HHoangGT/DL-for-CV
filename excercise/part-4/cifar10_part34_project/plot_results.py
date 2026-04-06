import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {k: float(v) if k != "epoch" else int(v) for k, v in row.items()}
            )
    return rows


def plot_metric(run_dirs, metric_name, output_path):
    plt.figure(figsize=(8, 5))
    for run_dir in run_dirs:
        history = load_history(Path(run_dir) / "history.csv")
        epochs = [row["epoch"] for row in history]
        values = [row[metric_name] for row in history]
        plt.plot(epochs, values, marker="o", label=Path(run_dir).name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path("results") / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_time_sec"]
    for metric in metrics:
        plot_metric(args.runs, metric, out_dir / f"{metric}.png")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
