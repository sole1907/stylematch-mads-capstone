"""Hyperparameter iteration #2: a narrow follow-up to the real
full-scale sweep (job 66002496, see reports/staged_search_stage_{1,2,3}.csv
and reports/staged_search_summary.csv), triggered specifically because
that sweep's margin winner (0.5) sat at the literal edge of the tested
range ([0.05, 0.1, 0.15, 0.2, 0.3, 0.5]), so the true optimum may lie
outside it.

learning_rate=1e-5 and freeze_until_block=10 are fixed at the original
sweep's confirmed winners, not re-searched here: the original design
already treated margin as only weakly coupled to those two (that's why
it was its own separate, later stage to begin with), so there's no
need to re-open that question.

num_steps is swept here too ([30, 60, 100, 200, 500], the same grid as
the original sweep's stage 3), not fixed at 60: a bigger margin keeps
more triplets "unsatisfied" for longer (TripletMarginLoss zeroes a
triplet's gradient once d(neg) - d(pos) > margin), so margin=0.7/1.0
could plausibly peak at a different num_steps than margin=0.5's peak
of 60. Fixing num_steps=60 here would unfairly undersell the new
margins relative to 0.5, which already got its own dedicated num_steps
search to find its peak.

Only margin=[0.7, 1.0] are run here, not 0.5: the full
learning_rate=1e-5, freeze_until_block=10, margin=0.5 row across all 5
num_steps values already exists in reports/staged_search_stage_3.csv
(peak improvement=+0.1167 at num_steps=60), no need to re-pay for data
already in hand. That existing row is the baseline the ten new ones
here get compared against.

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
        num_steps=[30, 60, 100, 200, 500],
        num_categories=None,
        max_images_per_category=None,
    )
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["improvement"].idxmax()]
    print(
        "Best (margin, num_steps) tested here:",
        (best_row["margin"], best_row["num_steps"]),
        "improvement:",
        best_row["improvement"],
    )
    print("Compare against margin=0.5 baseline (staged_search_stage_3.csv): peak improvement=+0.1167 at num_steps=60")


if __name__ == "__main__":
    main()
