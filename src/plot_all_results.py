"""
plot_all_results.py
Figure 7: comprehensive comparison of all methods across all experiments.
"""

from __future__ import annotations
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES = Path("../results/figures")

def load_all() -> pd.DataFrame:
    """Load and merge all benchmark results."""
    frames = []

    # Main benchmark (TEASER + ECONOMY-K)
    try:
        df = pd.read_csv("../results/benchmark_results_full.csv")
        frames.append(df[["dataset","method","accuracy","earliness","hm_score"]])
    except FileNotFoundError:
        pass

    # Two-threshold orchestration
    try:
        df2 = pd.read_csv("../results/orchestrator2_results.csv")
        frames.append(df2[["dataset","method","accuracy","earliness","hm_score"]])
    except FileNotFoundError:
        pass

    if not frames:
        raise FileNotFoundError("No result files found. Run benchmarks first.")

    df = pd.concat(frames, ignore_index=True)
    # Remove duplicate ECONOMY-K rows
    df = df.drop_duplicates(subset=["dataset","method"]).reset_index(drop=True)
    return df


def plot_comprehensive(df: pd.DataFrame) -> None:
    """
    Figure 7: 2x2 panel showing the full experimental landscape.
    Panel A: Accuracy vs Earliness scatter (all methods)
    Panel B: Mean HM Score bar chart (all methods)
    Panel C: Per-dataset HM heatmap
    Panel D: Trilemma triangle (earliness, accuracy, stability proxy)
    """

    # Colour scheme by method family
    COLORS = {
        "TEASER":           "#2563EB",
        "ECONOMY-K":        "#DC2626",
        "EK-Aggressive":    "#F97316",
        "EK-Conservative":  "#7C3AED",
        "Orch2-OR":         "#059669",
        "Orch2-AND":        "#0891B2",
        "Orch2-WEIGHTED":   "#D97706",
    }
    MARKERS = {
        "TEASER":           "o",
        "ECONOMY-K":        "s",
        "EK-Aggressive":    "^",
        "EK-Conservative":  "v",
        "Orch2-OR":         "D",
        "Orch2-AND":        "P",
        "Orch2-WEIGHTED":   "X",
    }
    DEFAULT_COLOR  = "#6B7280"
    DEFAULT_MARKER = "o"

    methods = df["method"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Stream Anomaly Benchmark — Full Experimental Landscape\n"
        "TEASER vs ECONOMY-K vs Two-Threshold Orchestration",
        fontsize=14, fontweight="bold"
    )

    # Panel A: Pareto scatter
    ax = axes[0, 0]
    for method in methods:
        grp = df[df["method"] == method]
        ax.scatter(
            grp["earliness"], grp["accuracy"],
            c=COLORS.get(method, DEFAULT_COLOR),
            marker=MARKERS.get(method, DEFAULT_MARKER),
            s=80, label=method, alpha=0.85, zorder=3,
        )
    ax.set_xlabel("Earliness", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("A — Accuracy vs Earliness (all methods)", fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    # Panel B: Mean HM Score bar chart
    ax = axes[0, 1]
    mean_hm = df.groupby("method")["hm_score"].mean().sort_values(ascending=True)
    colors_bar = [COLORS.get(m, DEFAULT_COLOR) for m in mean_hm.index]
    bars = ax.barh(mean_hm.index, mean_hm.values, color=colors_bar, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xlabel("Mean HM Score", fontsize=11)
    ax.set_title("B — Mean HM Score by Method", fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3)

    # Panel C: Per-dataset HM heatmap
    ax = axes[1, 0]
    pivot = df.pivot_table(
        index="dataset", columns="method", values="hm_score", aggfunc="mean"
    )
    # Keep only key methods for readability
    key_methods = ["TEASER","ECONOMY-K","EK-Aggressive","EK-Conservative",
                   "Orch2-OR","Orch2-AND","Orch2-WEIGHTED"]
    cols = [m for m in key_methods if m in pivot.columns]
    pivot = pivot[cols]

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "HM Score"},
    )
    ax.set_title("C — HM Score Heatmap (dataset × method)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Panel D: Trilemma radar
    ax = axes[1, 1]
    ax.set_aspect("equal")

    # Use 3 axes: accuracy, earliness, hm_score
    metrics = ["accuracy", "earliness", "hm_score"]
    labels  = ["Accuracy", "Earliness", "HM Score"]
    n       = len(metrics)
    angles  = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax_radar = fig.add_subplot(2, 2, 4, polar=True)
    axes[1, 1].remove()

    key_plot = ["TEASER","ECONOMY-K","EK-Aggressive","EK-Conservative",
                "Orch2-OR","Orch2-AND"]
    for method in key_plot:
        grp = df[df["method"]==method]
        if grp.empty:
            continue
        vals = grp[metrics].mean().tolist() + grp[metrics].mean().tolist()[:1]
        ax_radar.plot(
            angles, vals,
            color=COLORS.get(method, DEFAULT_COLOR),
            lw=1.8, label=method,
        )
        ax_radar.fill(
            angles, vals,
            color=COLORS.get(method, DEFAULT_COLOR),
            alpha=0.05,
        )

    ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("D — Trilemma Profile\n(mean across datasets)",
                       fontweight="bold", pad=20, fontsize=11)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / "fig7_comprehensive_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading all results...")
    df = load_all()
    print(f"  {len(df)} rows across {df['method'].nunique()} methods")
    print("\nGenerating Figure 7...")
    plot_comprehensive(df)
    print("Done.")