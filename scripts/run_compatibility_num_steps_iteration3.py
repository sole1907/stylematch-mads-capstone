"""Hyperparameter iteration #3: a further follow-up, triggered because
iteration #2's winner (margin=0.02, num_steps=2000, see
reports/compatibility_num_steps_iteration2.csv) had BOTH its values
sitting at the edge of what was tested there -- margin=0.02 was the
floor of every margin grid tried in this project so far, and
num_steps=2000 was the top of iteration #2's own range.

learning_rate=1e-5 and freeze_until_block=10 stay fixed at the
original sweep's stage 1 winners, same as iteration #2, for the same
reason: nothing about either edge calls that stage into question.

margin: the full iteration #2 grid, plus two smaller values
(0.005, 0.01) to push past its floor: [0.005, 0.01, 0.02, 0.05, 0.1,
0.15, 0.2, 0.25, 0.3].

num_steps: [2000, 3000, 4000] -- 2000 kept as the continuity point
back to iteration #2's winner, 1000 dropped (already conclusively
worse there, no value in re-confirming it), 3000/4000 pushing past the
prior edge.

Fresh standalone results file (reports/compatibility_num_steps_iteration3.csv),
not resuming from iteration #2's file: a deliberate choice for a clean,
self-contained record of this iteration, even though a few
combinations here (margin=0.02 at num_steps=2000) were already
computed there.

Same full-scale evaluation as the real sweep (no FITB/compatibility
eval cap), not a reduced subset.

Run directly:
    python scripts/run_compatibility_num_steps_iteration3.py
"""

from stylematch.compatibility.sweep import build_grid, run_sweep
from stylematch.config import REPORTS_DIR

RESULTS_PATH = REPORTS_DIR / "compatibility_num_steps_iteration3.csv"


def main() -> None:
    configs = build_grid(
        num_steps=[2000, 3000, 4000],
        margin=[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
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
        "Compare against iteration #2's winner "
        "(compatibility_num_steps_iteration2.csv): margin=0.02, num_steps=2000, "
        "auc_improvement=+0.202"
    )


if __name__ == "__main__":
    main()
