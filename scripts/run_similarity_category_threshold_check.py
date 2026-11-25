"""Follow-up check for the similarity engine: does dropping thin
categories (under 500 images) from training and evaluation entirely
improve retrieval accuracy, compared to keeping all 46 qualifying
categories -- and, combined with that, does category-size-weighted
sampling behave differently once there's no longer a long tail of
thin, fast-saturating categories to begin with?

Triggered by the real per-category distribution (46 qualifying
categories, 11-52,138 images each, 22 of them under 500 images) and by
run_similarity_sampling_check.py's result: weighting on its own
consistently underperformed unweighted sampling at every tested
setting. Rather than conclude weighting never helps, this re-includes
it alongside the threshold question: it's possible weighting only hurt
because thin categories were still competing for attention in its
later rounds, and dropping them changes that.

learning_rate=1e-5, freeze_until_block=10, margin=0.7 stay fixed at
the already-confirmed winners.

Grid: category_inclusion_threshold=[None, 500] x
weight_by_category_size=[False, True] x num_steps=[100, 200, 500] x
images_per_category=[4, 8, 16] -- 36 configs. None keeps all 46
categories (the baseline to compare against); 500 drops the 22 thin
ones, keeping 24. Both before/after evaluation use the same threshold
as training, so a config's own eval is always measured against the
same category pool it actually trained on.

Results saved to their own file
(reports/similarity_category_threshold_check.csv), never touching any
earlier sweep/iteration/check's results.

Run directly:
    python scripts/run_similarity_category_threshold_check.py
"""

from itertools import product

from stylematch.config import REPORTS_DIR
from stylematch.similarity.sweep import TrainingConfig, run_sweep

RESULTS_PATH = REPORTS_DIR / "similarity_category_threshold_check.csv"


def main() -> None:
    configs = [
        TrainingConfig(
            learning_rate=1e-5,
            freeze_until_block=10,
            margin=0.7,
            num_categories=None,
            weight_by_category_size=weighted,
            category_inclusion_threshold=threshold,
            num_steps=steps,
            images_per_category=images,
        )
        for threshold, weighted, steps, images in product(
            [None, 500], [False, True], [100, 200, 500], [4, 8, 16]
        )
    ]
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print(
        "Best (category_inclusion_threshold, weight_by_category_size, num_steps, "
        "images_per_category):",
        (
            best_row["category_inclusion_threshold"],
            best_row["weight_by_category_size"],
            best_row["num_steps"],
            best_row["images_per_category"],
        ),
        "improvement:",
        best_row["improvement"],
    )
    print(
        "Compare against the sampling check's winner "
        "(all 46 categories, unweighted, num_steps=500, images_per_category=16): improvement=+0.161"
    )


if __name__ == "__main__":
    main()
