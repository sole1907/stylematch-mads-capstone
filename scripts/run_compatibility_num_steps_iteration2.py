"""Hyperparameter iteration #2: a narrow follow-up to the real
full-scale compatibility sweep (see
reports/compatibility_staged_search_stage_{1,2,3}.csv and
reports/compatibility_staged_search_summary.csv), triggered
specifically because that sweep's num_steps winner (500) sat at the
literal edge of the tested range ([30, 60, 100, 200, 500]), so the
true optimum may lie outside it.

learning_rate=1e-5 and freeze_until_block=10 are fixed at the original
sweep's confirmed stage 1 winners, not re-searched here: nothing about
the num_steps edge calls that stage into question.

margin is swept here too (the full original grid, not fixed at 0.15):
that 0.15 winner was found with num_steps fixed at only 30 (stage 2
never varies num_steps), so a longer training run could plausibly
favor a different margin than what 30 steps found best. Fixing
margin=0.15 here would unfairly undersell the new num_steps values
relative to whatever margin actually suits them.

Only num_steps=[1000, 2000] are run here, not 500: margin=0.15 across
all 5 original num_steps values already exists in
reports/compatibility_staged_search_stage_3.csv (improvement=+0.1493
at num_steps=500), no need to re-pay for data already in hand. That
existing row is the baseline the new rows here get compared against.

Same full-scale evaluation as the real sweep (no FITB/compatibility
eval cap), not a reduced subset.

Results saved to their own file
(reports/compatibility_num_steps_iteration2.csv), never overwriting
the original sweep's already-committed
reports/compatibility_staged_search_stage_{2,3}.csv.

Run directly:
    python scripts/run_compatibility_num_steps_iteration2.py
"""

from stylematch.compatibility.sweep import build_grid, run_sweep
from stylematch.config import REPORTS_DIR

RESULTS_PATH = REPORTS_DIR / "compatibility_num_steps_iteration2.csv"


def main() -> None:
    configs = build_grid(
        num_steps=[1000, 2000],  # 500 already exists in compatibility_staged_search_stage_3.csv
        margin=[0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        learning_rate=[1e-5],
        freeze_until_block=[10],
    )
    results = run_sweep(configs, results_path=RESULTS_PATH)
    print(results)
    best_row = results.loc[results["auc_improvement"].idxmax()]
    print(
        "Best (margin, num_steps) tested here:",
        (best_row["margin"], best_row["num_steps"]),
        "auc_improvement:",
        best_row["auc_improvement"],
    )
    print(
        "Compare against margin=0.15 baseline "
        "(compatibility_staged_search_stage_3.csv): improvement=+0.1493 at num_steps=500"
    )


if __name__ == "__main__":
    main()
