"""Follow-up check for the similarity engine: does category-size-
weighted batch sampling (weight_by_category_size, see sampler.py)
actually improve retrieval accuracy, and does the original sweep's
num_steps winner (100) still hold once the sampler's per-epoch
behavior changes?

Triggered by a closer look at the real per-category image
distribution (46 qualifying categories, 11-52,138 images each): the
original sweep always used weight_by_category_size=False and
images_per_category=4 -- neither was ever varied, so neither
contribution is actually known yet.

learning_rate=1e-5, freeze_until_block=10, margin=0.7 stay fixed at
the original sweep's (plus margin_iteration2's) confirmed winners --
nothing about this check calls those into question.

Grid: weight_by_category_size=[False, True] x num_steps=[100, 200,
500] x images_per_category=[4, 8, 16] -- 18 configs. images_per_category
values checked beforehand against the real data: 4 and 8 both keep all
46 categories, 16 drops only 2 (Halter, Coverup; 24 images total,
negligible).

Results saved to their own file
(reports/similarity_sampling_check.csv), never touching the original
sweep's reports/staged_search_stage_*.csv or
reports/margin_iteration2.csv.

Run directly:
    python scripts/run_similarity_sampling_check.py
"""

from itertools import product

from stylematch.config import REPORTS_DIR
from stylematch.similarity.sweep import TrainingConfig, run_sweep

RESULTS_PATH = REPORTS_DIR / "similarity_sampling_check.csv"


def main() -> None:
    configs = [
        TrainingConfig(
            learning_rate=1e-5,
            freeze_until_block=10,
            margin=0.7,
            num_categories=None,
            weight_by_category_size=weighted,
            num_steps=steps,
            images_per_category=images,
        )
        for weighted, steps, images in product([False, True], [100, 200, 500], [4, 8, 16])
    ]
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print(
        "Best (weight_by_category_size, num_steps, images_per_category):",
        (best_row["weight_by_category_size"], best_row["num_steps"], best_row["images_per_category"]),
        "improvement:",
        best_row["improvement"],
    )
    print(
        "Compare against the original sweep's winner "
        "(unweighted, num_steps=100, images_per_category=4): improvement=+0.167"
    )


if __name__ == "__main__":
    main()
