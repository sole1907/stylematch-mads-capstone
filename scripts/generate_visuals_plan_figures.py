"""Rough-draft figures for the Visuals Mini-Deliverable (SIADS 699).

Three of the five figures use real, already-committed results (category
distribution computed from the real DeepFashion annotations, the naive-
vs-trained baseline comparison from Checklist-documented numbers, and the
margin x num_steps heatmap from reports/staged_search_stage_3.csv +
margin_iteration2.csv). The last two (compatibility engine) are explicitly
mockups: that component hasn't been built yet, so they're drawn with
illustrative, clearly-labeled placeholder data, not measured results.
Figure 4 sketches a generic compatibility-score comparison; figure 5
sketches FITB accuracy and ROC-AUC specifically, the standard
compatibility-task metrics (Vasileva et al.) this project already
committed to using — and includes Vasileva et al.'s real published
numbers (verified via the paper/its repo, not quoted from memory) as
a reference point alongside the fabricated "ours" placeholder, with
the two kept visually distinct.

Every figure has two header lines: a small grey "eyebrow" naming what
the figure is (Rule 1: know your audience — a skimming reader should
be able to tell what they're looking at before reading the takeaway),
and a bold headline stating the one thing it's meant to prove (Rule 2:
identify the message).

Other design choices follow Rougier, Droettboom & Fleet (2014), "Ten
Simple Rules for Better Figures" (PLOS Comp Bio):
  - bar charts start at 0, values are labeled directly on the bars
    rather than requiring the reader to read off an axis (Rule 9: don't
    mislead; Rule 5: avoid chartjunk)
  - figure 1 deliberately avoids a log-scale axis: compressing a
    4,739x range onto one log axis made bar length stop reflecting
    real size (a 5x-larger bar could look barely longer), exactly the
    "data doesn't reflect proper size" failure mode the article warns
    about. Two linear panels (full range + a zoomed tail) instead, so
    bar length is always directly proportional to count (Rule 9)
  - the heatmap uses a diverging, colorblind-safe colormap centered on
    zero (not red/green) and annotates every cell with its number, with
    text/border color picked from each cell's actual rendered luminance
    so the darkest cell (the winner) stays legible (Rule 6: use color
    effectively)
  - the mockup figure is visually distinct (hatched, watermarked,
    labeled "ILLUSTRATIVE") so it can't be mistaken for a real result,
    and its ROC curve's AUC is computed from the curve actually drawn,
    not asserted separately, so the figure can't contradict itself
    (Rule 9: don't mislead the reader)

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
EYEBROW_COLOR = "#8a8a8a"
HEADLINE_COLOR = "#222222"


def add_header(fig, eyebrow: str, headline: str, eyebrow_y: float, headline_y: float, headline_fontsize: float = 11) -> None:
    """Small grey label naming the figure, then a bold headline stating
    its one takeaway, both left-aligned at the top of the figure."""
    fig.text(0.02, eyebrow_y, eyebrow.upper(), fontsize=8.5, color=EYEBROW_COLOR, ha="left", fontweight="bold")
    fig.text(0.02, headline_y, headline, fontsize=headline_fontsize, color=HEADLINE_COLOR, ha="left", fontweight="bold")


def figure_1_category_distribution() -> None:
    """Real data: per-category image counts in the train split.

    Two linear-scale panels, not one log-scale panel: a log axis
    compresses the gap between e.g. 10,000 and 52,138 so much that a
    5x-larger bar can look barely longer than its neighbor — exactly
    the "data doesn't reflect proper size" problem the article warns
    about, and confusing for a reader who doesn't already think in
    exponents. Left panel is the honest full-range picture (bar length
    truly proportional to count, most of the tail is barely visible —
    that IS the real story); right panel re-plots just the tail
    (everything under 3,000 images) on its own linear scale so those
    smaller categories' real differences are visible too.
    """
    ds = DeepFashionCAPDataset(split="train", min_images_per_category=4, num_categories=None, max_images_per_category=None)
    counts = pd.Series([s[1] for s in ds.samples]).value_counts()

    with open(REPORTS_DIR.parent / "data" / "raw" / "deepfashion" / "list_category_cloth.txt") as f:
        lines = f.readlines()[2:]
    cat_names = {i: line.split()[0] for i, line in enumerate(lines)}

    counts = counts.rename(index=cat_names).sort_values(ascending=True)

    zoom_cutoff = 3000
    zoomed = counts[counts < zoom_cutoff]

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11, 9.3), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    ax_full.barh(counts.index, counts.values, color=ACCENT, height=0.7)
    ax_full.set_xlabel("Images in train split (linear)")
    ax_full.set_ylabel("DeepFashion garment category")
    ax_full.set_title("All 46 categories", fontsize=10, loc="left")
    ax_full.tick_params(axis="y", labelsize=7)
    ax_full.spines["left"].set_visible(False)
    ax_full.annotate(
        f"{counts.idxmax()}: {counts.max():,}",
        xy=(counts.max(), len(counts) - 1),
        xytext=(-6, 0), textcoords="offset points", va="center", ha="right",
        fontsize=8, color="white", fontweight="bold",
    )
    # Mark, on the full-range panel, exactly which categories the zoomed
    # panel re-plots — so a reader isn't left wondering why the right
    # panel's bars don't match what they see here.
    ax_full.axhline(len(zoomed) - 0.5, color="#555555", linewidth=1, linestyle=":")
    ax_full.annotate(
        f"↓ {len(zoomed)} categories under {zoom_cutoff:,} images,\nzoomed in the right panel →",
        xy=(counts.max() * 0.35, len(zoomed) / 2), fontsize=8, color="#555555", style="italic",
    )

    ax_zoom.barh(zoomed.index, zoomed.values, color=ACCENT, height=0.7)
    ax_zoom.set_xlabel("Images in train split (linear, zoomed)")
    ax_zoom.set_title(f"Zoomed: the {len(zoomed)} categories under {zoom_cutoff:,} images", fontsize=10, loc="left")
    ax_zoom.tick_params(axis="y", labelsize=7)
    ax_zoom.spines["left"].set_visible(False)
    ax_zoom.annotate(
        f"{zoomed.idxmin()}: {zoomed.min()}",
        xy=(zoomed.min(), 0),
        xytext=(6, 0), textcoords="offset points", va="center",
        fontsize=8, color=ACCENT,
    )

    add_header(
        fig,
        eyebrow="Figure 1 — DeepFashion category sizes",
        headline=f"Category sizes span a {counts.max() // counts.min()}x range — this is why batches must\n"
        "be built category-balanced, not randomly",
        eyebrow_y=0.975, headline_y=0.935,
    )
    fig.text(
        0.02, 0.01,
        "Figure 1. Real image counts per DeepFashion category (train split, the 46 categories qualifying with "
        f">=4 images). Range: {counts.min()}-{counts.max():,} images, a {counts.max() // counts.min()}x imbalance. "
        "Both panels are linear scale — no log axis — so bar length is always directly proportional to count.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(FIGURES_DIR / "01_category_distribution.png", bbox_inches="tight")
    plt.close(fig)


def figure_2_baseline_comparison() -> None:
    """Real data: random vs. naive color-histogram vs. trained model,
    same metric, same tiny validation subset (see Checklist). That
    subset is also the one used to pick the hyperparameters shown here
    (learning_rate=1e-4, freeze_until_block=8 — the small-subset
    winner), which the full-scale sweep later found did NOT hold at
    scale (see dev-notes). So this comparison is somewhat optimistic in
    the trained model's favor: it's evaluated on the same data that
    picked its hyperparameters, not a held-out set."""
    methods = ["Random\nchance", "Naive color\nhistogram", "Trained similarity\nmodel (ours)"]
    accuracy = [3.8, 18.75, 53.75]
    colors = [NEUTRAL, NEUTRAL, ACCENT]

    fig, ax = plt.subplots(figsize=(6, 6.2))
    bars = ax.bar(methods, accuracy, color=colors, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 retrieval accuracy (%)")
    add_header(
        fig,
        eyebrow="Figure 2 — similarity model vs. naive baseline",
        headline="Learned embeddings beat a hand-crafted color baseline\nby >2.8x, not just random chance",
        eyebrow_y=0.975, headline_y=0.925,
    )
    fig.text(
        0.02, 0.825,
        "Caveat: these numbers used the small-subset-chosen hyperparameters (learning_rate=1e-4, "
        "freeze_until_block=8). The full-scale sweep later found the real winner (margin=0.7, num_steps=100, "
        "learning_rate=1e-5, freeze_until_block=10) — hyperparameter tuning itself is already done; what's "
        "pending is a one-time re-run of this same comparison using that final config, not more tuning.",
        fontsize=8, color="#a05a2c", ha="left", wrap=True, style="italic",
    )
    for bar, val in zip(bars, accuracy):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    fig.text(
        0.02, 0.01,
        "Figure 2. Same stratified validation subset, same metric (top-1 nearest-neighbor by category). "
        "Will be recomputed on the full-scale final config (margin=0.7, num_steps=100) before the final report.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.8))
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

    fig, ax = plt.subplots(figsize=(7, 4.6))
    vmax = combined["improvement"].abs().max()
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("num_steps")
    ax.set_ylabel("margin")

    peak_val = pivot.values.max()
    cmap = plt.get_cmap("RdBu")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            is_peak = val == peak_val
            # Pick white or black text based on this cell's actual rendered
            # color, not a fixed guess — dark navy cells (strong positive
            # improvement, like the peak) need white text; pale cells need
            # black, so every cell stays legible regardless of position on
            # the colormap.
            normalized = 0.5 + 0.5 * (val / vmax) if vmax else 0.5
            r, g, b, _ = cmap(normalized)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.6 else "black"
            ax.text(
                j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold" if is_peak else "normal",
                color=text_color,
            )
            if is_peak:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=text_color, lw=2.5)
                )

    add_header(
        fig,
        eyebrow="Figure 3 — hyperparameter tuning: margin x num_steps",
        headline="Different margins peak at different num_steps —\nmargin=0.7 wants 100 steps, margin=0.5 wants 60",
        eyebrow_y=0.97, headline_y=0.895,
    )
    fig.colorbar(im, ax=ax, label="improvement (top-1 accuracy, after - before)", shrink=0.8)
    fig.text(
        0.02, 0.01,
        "Figure 3. Real results, Great Lakes jobs 66002496 + 66049320, drawn from all 46 real categories with "
        "no per-category image cap (not a capped subset) — each short run still only samples a small slice of "
        "that pool per step, not every image at once. Boxed cell = best config found: margin=0.7, num_steps=100, "
        "improvement=+0.167.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.8))
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

    fig, ax = plt.subplots(figsize=(6, 5.3))
    ax.hist(random_pairs, bins=25, alpha=0.5, color=NEUTRAL, label="Random outfit pairs", hatch="//")
    ax.hist(true_pairs, bins=25, alpha=0.5, color=ACCENT, label="Real outfit pairs", hatch="//")
    ax.set_xlabel("Compatibility score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    add_header(
        fig,
        eyebrow="Figure 4 (planned) — compatibility score, true vs. random pairs",
        headline="PLANNED: real outfit pairs should score higher than random pairs",
        eyebrow_y=0.96, headline_y=0.9,
    )
    fig.text(
        0.02, 0.845,
        "What this compares, once built: the compatibility score for items actually worn together in a real "
        "outfit vs. items paired at random — to check the model tells coordinated outfits apart from mismatched "
        "ones. Illustrative, not real data.",
        fontsize=8.5, color="#444444", ha="left", wrap=True,
    )
    fig.text(
        0.5, 0.45, "ILLUSTRATIVE — MOCKUP\nNOT REAL DATA",
        fontsize=22, color="#cccccc", ha="center", va="center", rotation=30,
        alpha=0.6, fontweight="bold", transform=ax.transAxes,
    )
    fig.text(
        0.02, 0.01,
        "Figure 4 (planned). Compatibility engine not yet built. This sketches the intended figure once it "
        "exists: a compatibility-score histogram for true vs. random outfit pairs. Numbers shown are fabricated.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.78))
    fig.savefig(FIGURES_DIR / "04_compatibility_engine_mockup.png", bbox_inches="tight")
    plt.close(fig)


def figure_5_compatibility_engine_metrics_mockup() -> None:
    """PARTLY REAL, PARTLY ILLUSTRATIVE: the compatibility engine hasn't
    been built yet, so "ours" is fabricated — but Vasileva et al.
    (2018, ECCV), "Learning Type-Aware Embeddings for Fashion
    Compatibility", published real FITB/AUC numbers on the Polyvore
    Outfits dataset (nondisjoint split): AUC=0.86, FITB=55.3%. Verified
    via https://arxiv.org/abs/1803.09196 and the paper's own repo
    (https://github.com/mvasil/fashion-compatibility), not quoted from
    memory. Included as a real reference point, not just random
    chance, so "ours" has a literature bar to clear, not just a floor.
    The real (Vasileva) and fabricated ("ours") elements are kept
    visually distinct — only "ours" is hatched and annotated
    "(illustrative)" — so the one real citation in this mockup isn't
    swept into the same "not real data" bucket as the placeholder
    numbers."""
    VASILEVA_FITB = 55.3
    VASILEVA_AUC = 0.86
    CITED = "#8a6d3b"  # distinct from NEUTRAL/ACCENT: marks "real, cited" data specifically

    fig, (ax_fitb, ax_roc) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: FITB accuracy — random guess, the real Vasileva et al.
    # published number, and our fabricated placeholder.
    fitb_methods = ["Random\nguess (1-of-4)", "Vasileva et al. 2018\n(published, cited)", "Compatibility model\n(ours, illustrative)"]
    fitb_acc = [25.0, VASILEVA_FITB, 65.0]
    colors = [NEUTRAL, CITED, ACCENT]
    hatches = [None, None, "//"]
    bars = ax_fitb.bar(fitb_methods, fitb_acc, color=colors, width=0.6)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax_fitb.set_ylim(0, 100)
    ax_fitb.set_ylabel("FITB accuracy (%)")
    ax_fitb.set_title("Fill-in-the-blank accuracy", fontsize=10, loc="left")
    for bar, val in zip(bars, fitb_acc):
        ax_fitb.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                          textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    # Panel B: our illustrative ROC curve vs. the random diagonal.
    # Vasileva et al.'s AUC is cited as a reference number, not drawn
    # as a curve — their actual curve shape isn't something we have,
    # and fabricating one for a real citation would misrepresent it.
    fpr = np.linspace(0, 1, 200)
    tpr = np.power(fpr, 0.4)
    auc = np.trapezoid(tpr, fpr)
    ax_roc.plot(fpr, tpr, color=ACCENT, lw=2, linestyle=(0, (6, 2)), label=f"Ours, illustrative (AUC={auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, lw=1.5, linestyle="--", label="Random (AUC=0.50)")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve (compatible vs. incompatible)", fontsize=10, loc="left")
    ax_roc.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_roc.set_aspect("equal")
    ax_roc.text(
        0.04, 0.96,
        f"Reference: Vasileva et al. 2018\nreported AUC={VASILEVA_AUC} (published).\nCurve shape not published —\nnumber only, not plotted.",
        fontsize=7.5, color=CITED, va="top", ha="left", transform=ax_roc.transAxes,
    )

    add_header(
        fig,
        eyebrow="Figure 5 (planned) — compatibility engine evaluation metrics",
        headline="PLANNED: fill-in-the-blank accuracy and ROC-AUC, the standard outfit-\n"
        "compatibility metrics (Vasileva et al.) — \"ours\" is illustrative, Vasileva's bar is real",
        eyebrow_y=0.97, headline_y=0.885,
    )
    fig.text(
        0.02, 0.01,
        "Figure 5 (planned). Compatibility engine not yet built. FITB = pick the correct missing item from "
        "4 candidates (random guess = 25%); ROC-AUC = compatible-vs-incompatible pair classification. Vasileva et "
        "al. 2018 (Polyvore Outfits, nondisjoint split) is a real published citation, not fabricated — only the "
        "hatched \"ours\" bar and dashed \"ours\" curve are fabricated placeholders.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.78))
    fig.savefig(FIGURES_DIR / "05_compatibility_engine_metrics_mockup.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_category_distribution()
    figure_2_baseline_comparison()
    figure_3_margin_numsteps_heatmap()
    figure_4_compatibility_engine_mockup()
    figure_5_compatibility_engine_metrics_mockup()
    print(f"Saved 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
