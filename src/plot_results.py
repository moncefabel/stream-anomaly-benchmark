"""
plot_results.py
Generates the 4 main benchmark figures, saved to results/figures/.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"TEASER": "#2563EB", "ECONOMY-K": "#DC2626"}
MARKERS = {"TEASER": "o", "ECONOMY-K": "s"}
FIGURES = Path("../results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def load(path: str = "../results/benchmark_results_full.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # fallback to fast results if full not ready
    if not Path(path).exists():
        df = pd.read_csv("../results/benchmark_results.csv")
    return df


# Figure 1: Accuracy vs Earliness (Pareto scatter)

def plot_pareto(df: pd.DataFrame) -> None:
    """Accuracy vs earliness scatter; each point is one (dataset, method) pair."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, grp in df.groupby("method"):
        ax.scatter(
            grp["earliness"], grp["accuracy"],
            c=COLORS[method], marker=MARKERS[method],
            s=90, label=method, zorder=3, alpha=0.85,
        )
        # Annotate dataset names
        for _, row in grp.iterrows():
            ax.annotate(
                row["dataset"][:12],
                (row["earliness"], row["accuracy"]),
                fontsize=7, alpha=0.7,
                xytext=(4, 3), textcoords="offset points",
            )

    ax.set_xlabel("Earliness  (1 = instant decision)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Accuracy–Earliness Trade-off\n(each point = one dataset)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.axline((0, 0), slope=1, color="gray", lw=0.8, ls="--", alpha=0.4,
               label="acc = earliness")
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = FIGURES / "fig1_pareto_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# Figure 2: HM Score bar chart

def plot_hm_bars(df: pd.DataFrame) -> None:
    """Grouped bar chart of HM score per dataset, coloured by method."""
    datasets = df["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(["TEASER", "ECONOMY-K"]):
        vals = [
            df[(df["dataset"] == d) & (df["method"] == method)]["hm_score"].values
            for d in datasets
        ]
        heights = [v[0] if len(v) > 0 else 0 for v in vals]
        bars = ax.bar(
            x + (i - 0.5) * width, heights, width,
            label=method, color=COLORS[method], alpha=0.85, zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [d[:14] for d in datasets], rotation=30, ha="right", fontsize=9
    )
    ax.set_ylabel("HM Score  (↑ better)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Harmonic Mean Score by Dataset\n"
        "HM = 2·(Accuracy·Earliness) / (Accuracy+Earliness)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "fig2_hm_score_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# Figure 3: Radar chart

def plot_radar(df: pd.DataFrame) -> None:
    """Radar chart of mean accuracy/earliness/HM score across all datasets."""
    metrics   = ["accuracy", "earliness", "hm_score"]
    labels    = ["Accuracy", "Earliness", "HM Score"]
    n_metrics = len(metrics)
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for method in ["TEASER", "ECONOMY-K"]:
        vals = df[df["method"] == method][metrics].mean().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[method], lw=2, label=method)
        ax.fill(angles, vals, color=COLORS[method], alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Global Method Profile\n(mean across all datasets)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    fig.tight_layout()
    out = FIGURES / "fig3_radar_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# Figure 4: Method wins heatmap

def plot_wins_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of which method wins per (dataset, metric)."""
    metrics  = ["accuracy", "earliness", "hm_score"]
    datasets = df["dataset"].unique()

    winner_map = np.zeros((len(datasets), len(metrics)))
    for i, d in enumerate(datasets):
        sub = df[df["dataset"] == d].set_index("method")
        for j, m in enumerate(metrics):
            if "TEASER" in sub.index and "ECONOMY-K" in sub.index:
                winner_map[i, j] = (
                    1 if sub.loc["TEASER", m] >= sub.loc["ECONOMY-K", m] else -1
                )

    fig, ax = plt.subplots(figsize=(7, max(4, len(datasets) * 0.55)))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        winner_map,
        ax=ax,
        xticklabels=["Accuracy", "Earliness", "HM Score"],
        yticklabels=[d[:16] for d in datasets],
        cmap=cmap, center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "← ECONOMY-K wins  |  TEASER wins →"},
        annot=np.where(
            winner_map == 1, "T", np.where(winner_map == -1, "E", "=")
        ),
        fmt="",
    )
    ax.set_title(
        "Head-to-Head: TEASER vs ECONOMY-K\n"
        "T = TEASER wins  |  E = ECONOMY-K wins",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGURES / "fig4_wins_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# Summary table

def print_summary(df: pd.DataFrame) -> None:
    print("\n=== MEAN METRICS BY METHOD ===\n")
    summary = df.groupby("method")[["accuracy", "earliness", "hm_score"]].mean()
    print(summary.round(4).to_string())

    print("\n=== WINS PER METHOD (HM Score) ===\n")
    datasets = df["dataset"].unique()
    teaser_wins = economy_wins = ties = 0
    for d in datasets:
        sub = df[df["dataset"] == d].set_index("method")
        if "TEASER" not in sub.index or "ECONOMY-K" not in sub.index:
            continue
        t = sub.loc["TEASER", "hm_score"]
        e = sub.loc["ECONOMY-K", "hm_score"]
        if t > e:
            teaser_wins += 1
        elif e > t:
            economy_wins += 1
        else:
            ties += 1
    print(f"  TEASER wins   : {teaser_wins}")
    print(f"  ECONOMY-K wins: {economy_wins}")
    print(f"  Ties          : {ties}")


# Main

if __name__ == "__main__":
    print("Loading results...")
    try:
        df = load("../results/benchmark_results_full.csv")
        print(f"  {len(df)} rows loaded from full benchmark.")
    except FileNotFoundError:
        df = load("../results/benchmark_results.csv")
        print(f"  {len(df)} rows loaded from fast benchmark (full not ready yet).")

    print("\nGenerating figures...")
    plot_pareto(df)
    plot_hm_bars(df)
    plot_radar(df)
    plot_wins_heatmap(df)
    print_summary(df)

    print("\nDone. All figures in results/figures/")