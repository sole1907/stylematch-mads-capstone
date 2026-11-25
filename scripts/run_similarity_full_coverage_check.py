"""Re-tests category_inclusion_threshold x images_per_category x
num_steps now that training uses real full-coverage epochs (sampler.py)
and shares one train-derived category set with evaluation (dataset.py's
allowed_categories), rather than independently re-deriving per split.

learning_rate=1e-5, freeze_until_block=10, margin=0.7 stay fixed --
already well-validated, not expected to depend on the coverage fix.

Grid: category_inclusion_threshold=[None, 500, 1000] x
images_per_category=[4, 8, 16] x num_steps=[100, 200, 500, 1000, 2000]
-- 45 configs.

Run directly:
    python scripts/run_similarity_full_coverage_check.py
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
        )
        for threshold, images, steps in product(
            [None, 500, 1000], [4, 8, 16], [100, 200, 500, 1000, 2000]
        )
    ]
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print("Best config:", best_row.to_dict())


if __name__ == "__main__":
    main()
