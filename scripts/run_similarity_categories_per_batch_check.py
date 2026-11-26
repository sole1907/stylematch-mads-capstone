"""categories_per_batch was never tested, always fixed at 4. Tests it
alongside the two leading images_per_category candidates.

Grid: category_inclusion_threshold=[500, 1000] x
images_per_category=[4, 8, 16] x num_steps=[200, 500, 1000] x
categories_per_batch=[4, 6, 8, 10, 12] -- 90 configs, reusing the 18
categories_per_batch=4 rows already in similarity_full_coverage_check.csv.

Run directly:
    python scripts/run_similarity_categories_per_batch_check.py
"""

from itertools import product

from stylematch.config import REPORTS_DIR
from stylematch.similarity.sweep import TrainingConfig, run_sweep

RESULTS_PATH = REPORTS_DIR / "similarity_full_coverage_check.csv"


def main() -> None:
    configs = [
        TrainingConfig(
            learning_rate=1e-5,
            freeze_until_block=10,
            margin=0.7,
            num_categories=None,
            category_inclusion_threshold=threshold,
            images_per_category=images,
            num_steps=steps,
            categories_per_batch=cpb,
        )
        for threshold, images, steps, cpb in product(
            [500, 1000], [4, 8, 16], [200, 500, 1000], [4, 6, 8, 10, 12]
        )
    ]
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print("Best config overall:", best_row.to_dict())


if __name__ == "__main__":
    main()
