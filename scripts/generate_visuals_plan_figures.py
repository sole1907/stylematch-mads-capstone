"""Rough-draft figures for the Visuals Mini-Deliverable (SIADS 699).

Three of the four figures use real, already-committed results (category
distribution computed from the real DeepFashion annotations, the naive-
vs-trained baseline comparison from Checklist-documented numbers, and the
margin x num_steps heatmap from reports/staged_search_stage_3.csv +
margin_iteration2.csv). The fourth (compatibility engine) is explicitly
a mockup: that component hasn't been built yet, so it's drawn with
illustrative, clearly-labeled placeholder data, not measured results.

Design choices follow Rougier, Droettboom & Fleet (2014), "Ten Simple
Rules for Better Figures" (PLOS Comp Bio):
  - every figure states its one takeaway in the title, not just what's
    plotted (Rule 2: identify the message)
  - bar charts start at 0, values are labeled directly on the bars
    rather than requiring the reader to read off an axis (Rule 9: don't
    mislead; Rule 5: avoid chartjunk)
  - the heatmap uses a diverging, colorblind-safe colormap centered on
    zero (not red/green) and annotates every cell with its number, so
    color is a redundant cue, not the only one (Rule 6: use color
    effectively)
  - the mockup figure is visually distinct (hatched, muted, labeled
    "ILLUSTRATIVE") so it can't be mistaken for a real result (Rule 1:
    know your audience; a reader skimming shouldn't come away thinking
    this is measured data)

Run directly:
    python scripts/generate_visuals_plan_figures.py
Outputs to reports/figures/ (gitignored, regenerate as needed).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stylematch.config import FIGURES_DIR, REPORTS_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)

ACCENT = "#2a6f97"  # the one meaningful accent color, reused consistently across figures
NEUTRAL = "#9aa5ad"


def figure_1_category_distribution() -> None:
    """Real data: per-category image counts in the train split."""
    ds = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=None, max_images_per_category=None)
    counts = pd.Series([s[1] for s in ds.samples]).value_counts()

    with open(REPORTS_DIR.parent / "data" / "raw" / "deepfashion" / "list_category_cloth.txt") as f:
        lines = f.readlines()[2:]
    cat_names = {i: line.split()[0] for i, line in enumerate(lines)}

    counts = counts.rename(index=cat_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(counts.index, counts.values, color=ACCENT, height=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Images in train split (log scale)")
    ax.set_title(
        f"Category sizes span a {counts.max() // counts.min()}x range —\n"
        "this is why batches must be built category-balanced, not randomly",
        fontsize=11,
        loc="left",
    )
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("left",):
        ax.spines[spine].set_visible(False)
    ax.annotate(
        f"{counts.idxmax()}: {counts.max():,}",
        xy=(counts.max(), len(counts) - 1),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=ACCENT,
    )
    ax.annotate(
        f"{counts.idxmin()}: {counts.min()}",
        xy=(counts.min(), 0),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=ACCENT,
    )
    fig.text(
        0.02, 0.01,
        "Figure 1. Real image counts per DeepFashion category (train split, the 46 categories qualifying with "
        f">=4 images). Range: {counts.min()}-{counts.max():,} images, a {counts.max() // counts.min()}x imbalance.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES_DIR / "01_category_distribution.png", bbox_inches="tight")
    plt.close(fig)


def figure_2_baseline_comparison() -> None:
    """Real data: random vs. naive color-histogram vs. trained model,
    same metric, same tiny validation subset (see Checklist)."""
    methods = ["Random\nchance", "Naive color\nhistogram", "Trained similarity\nmodel (ours)"]
    accuracy = [3.8, 18.75, 53.75]
    colors = [NEUTRAL, NEUTRAL, ACCENT]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(methods, accuracy, color=colors, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 retrieval accuracy (%)")
    ax.set_title(
        "Learned embeddings beat a hand-crafted color baseline\nby >2.8x, not just random chance",
        fontsize=11, loc="left",
    )
    for bar, val in zip(bars, accuracy):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    fig.text(
        0.02, 0.01,
        "Figure 2. Same stratified validation subset, same metric (top-1 nearest-neighbor by category). "
        "Small-subset numbers; will be recomputed on the full-scale final config (margin=0.7, num_steps=100) "
        "before the final report.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES_DIR / "02_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


def figure_3_margin_numsteps_heatmap() -> None:
    """Real data: margin x num_steps improvement, from the original
    sweep's stage 3 (margin=0.5 row) + margin iteration #2
    (margin=0.7, 1.0 rows)."""
    stage3 = pd.read_csv(REPORTS_DIR / "staged_search_stage_3.csv")
    iter2 = pd.read_csv(REPORTS_DIR / "margin_iteration2.csv")
    combined = pd.concat([stage3, iter2], ignore_index=True)

    pivot = combined.pivot(index="margin", columns="num_steps", values="improvement")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = combined["improvement"].abs().max()
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("margin")

    peak_val = pivot.values.max()
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            is_peak = val == peak_val
            ax.text(
                j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold" if is_peak else "normal",
                color="black",
            )
            if is_peak:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5))

    ax.set_title(
        "Different margins peak at different num_steps —\nmargin=0.7 wants 100 steps, margin=0.5 wants 60",
        fontsize=11, loc="left",
    )
    fig.colorbar(im, ax=ax, label="improvement (top-1 accuracy, after - before)", shrink=0.8)
    fig.text(
        0.02, 0.01,
        "Figure 3. Real results, full-scale (all 46 categories), Great Lakes jobs 66002496 + 66049320. "
        "Boxed cell = best config found: margin=0.7, num_steps=100, improvement=+0.167.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / "03_margin_numsteps_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def figure_4_compatibility_engine_mockup() -> None:
    """ILLUSTRATIVE ONLY: the compatibility engine hasn't been built
    yet. This sketches the intended figure shape (analogous to figure
    2's story, for the compatibility engine instead), using fabricated
    numbers, so it's visually distinct from the real figures above."""
    rng = np.random.default_rng(0)
    true_pairs = rng.normal(0.72, 0.1, 500).clip(0, 1)
    random_pairs = rng.normal(0.38, 0.15, 500).clip(0, 1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(random_pairs, bins=25, alpha=0.5, color=NEUTRAL, label="Random outfit pairs", hatch="//")
    ax.hist(true_pairs, bins=25, alpha=0.5, color=ACCENT, label="Real outfit pairs", hatch="//")
    ax.set_xlabel("Compatibility score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "PLANNED: real outfit pairs should score higher\nthan random pairs (illustrative, not real data)",
        fontsize=11, loc="left", color="#333333",
    )
    fig.text(
        0.5, 0.5, "ILLUSTRATIVE — MOCKUP\nNOT REAL DATA",
        fontsize=22, color="#cccccc", ha="center", va="center", rotation=30,
        alpha=0.6, fontweight="bold", transform=ax.transAxes,
    )
    fig.text(
        0.02, 0.01,
        "Figure 4 (planned). Compatibility engine not yet built. This sketches the intended figure once it "
        "exists: a compatibility-score histogram for true vs. random outfit pairs. Numbers shown are fabricated.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES_DIR / "04_compatibility_engine_mockup.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_category_distribution()
    figure_2_baseline_comparison()
    figure_3_margin_numsteps_heatmap()
    figure_4_compatibility_engine_mockup()
    print(f"Saved 4 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
