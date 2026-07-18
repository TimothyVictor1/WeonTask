"""Overlay several finished runs on one chart: the report's hero figure.

Every run must already have metrics.csv (run analyze_run.py on it first).

Usage:
    python src/compare_runs.py --runs data/runs/A data/runs/B data/runs/C \
        --labels plain preserve crop
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def read(run):
    with open(Path(run) / "metrics.csv") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="data/runs/comparison.png")
    args = ap.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    colors = [cm.tab10(i % 10) for i in range(len(args.runs))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    print(f"{'arm':<22}{'final SSIM':>12}{'final dE':>10}{'collateral%':>13}")
    for run, label, color in zip(args.runs, labels, colors):
        rows = read(run)
        xs = range(1, len(rows) + 1)
        axes[0].plot(xs, [float(r["cum_ssim"]) for r in rows],
                     marker="o", color=color, label=label)
        axes[1].plot(xs, [float(r["cum_collateral_pct"]) for r in rows],
                     marker="s", color=color, label=label)
        last = rows[-1]
        print(f"{label:<22}{last['cum_ssim']:>12}{last['cum_delta_e']:>10}"
              f"{last['cum_collateral_pct']:>13}")

    axes[0].set_title("Structure preserved outside edits (SSIM)")
    axes[0].set_xlabel("edits applied")
    axes[0].set_ylabel("SSIM vs original")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("% of untouched pixels visibly changed")
    axes[1].set_xlabel("edits applied")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Degradation across strategies, same base image and edits")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()