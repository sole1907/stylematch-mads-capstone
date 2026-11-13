"""Hyperparameter iteration #2: a narrow follow-up to the real
full-scale sweep (see dev-notes/full-scale-hyperparameter-sweep.md),
triggered specifically because that sweep's margin winner (0.5) sat at
the literal edge of the tested range ([0.05, 0.1, 0.15, 0.2, 0.3, 0.5]),
so the true optimum may lie outside it.

learning_rate=1e-5 and freeze_until_block=10 are fixed at the original
sweep's confirmed winners, not re-searched here: the original design
already treated margin as only weakly coupled to those two (that's why
it was its own separate, later stage to begin with), so there's no
need to re-open that question.

num_steps=60, not some other value: that's the actual overall winning
step count from the original sweep's stage 3 (the single best result
across the whole sweep, see staged_search_stage_3.csv), so this
compares new margins at the step count we'd actually use, not an
arbitrary one. (A separate "does a short-duration preference hold up
longer" question doesn't need its own check here: stage 3 already
tested this exact combination at num_steps up to 500 and found a
gentle decline past 60, not a reversal, so that's already answered by
existing data, not something this follow-up needs to re-test.)

Only margin=[0.7, 1.0] are run here, not 0.5: the combination
learning_rate=1e-5, freeze_until_block=10, margin=0.5, num_steps=60
already exists in reports/staged_search_stage_3.csv (improvement=
+0.1167, the single best result in the whole original sweep), no need
to re-pay for a data point already in hand. That existing row is the
baseline the two new ones here get compared against.

Same full-scale dataset as the real sweep (num_categories=None,
max_images_per_category=None), not a small subset.

Results saved to their own file (reports/margin_iteration2.csv), never
overwriting the original sweep's already-committed
reports/staged_search_stage_2.csv or staged_search_stage_3.csv.

Run directly:
    python scripts/run_similarity_margin_iteration2.py
"""

from stylematch.config import REPORTS_DIR
from stylematch.similarity.sweep import build_grid, run_sweep

RESULTS_PATH = REPORTS_DIR / "margin_iteration2.csv"


def main() -> None:
    configs = build_grid(
        margin=[0.7, 1.0],  # 0.5 already exists in staged_search_stage_3.csv, not re-run
        learning_rate=[1e-5],
        freeze_until_block=[10],
        num_steps=[60],
        num_categories=None,
        max_images_per_category=None,
    )
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print("Best margin tested here:", best_row["margin"], "improvement:", best_row["improvement"])
    print("Compare against margin=0.5 baseline (staged_search_stage_3.csv): improvement=+0.1167")


if __name__ == "__main__":
    main()
