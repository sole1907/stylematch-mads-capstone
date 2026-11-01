"""Tests for the hyperparameter sweep utility. build_grid's tests are
pure combinatorics, no dataset needed. run_sweep's test actually trains,
so it requires the DeepFashion CAP subset to be downloaded (skipped
otherwise), and uses a tiny config (few steps) to check the mechanics
work, not to produce a meaningful comparison, that's what sweep.py's
own __main__ block is for.
"""

import pytest

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.sweep import TrainingConfig, build_grid, run_sweep, run_staged_search


def test_build_grid_produces_full_cartesian_product():
    configs = build_grid(
        learning_rate=[1e-3, 1e-4],
        num_steps=[10, 20],
        margin=[0.1],
        freeze_until_block=[6, 8, 10],
    )
    # 2 learning_rate x 2 num_steps x 1 margin x 3 freeze_until_block
    assert len(configs) == 2 * 2 * 1 * 3
    assert all(isinstance(c, TrainingConfig) for c in configs)
    # No two configs should be identical combinations
    combos = {(c.learning_rate, c.num_steps, c.margin, c.freeze_until_block) for c in configs}
    assert len(combos) == len(configs)


def test_build_grid_passes_through_fixed_kwargs():
    configs = build_grid(learning_rate=[1e-4], num_categories=10)
    assert all(c.num_categories == 10 for c in configs)


requires_deepfashion = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


@requires_deepfashion
def test_run_sweep_produces_one_row_per_config(tmp_path):
    configs = [
        TrainingConfig(num_steps=2, num_categories=10),
        TrainingConfig(num_steps=2, num_categories=10, learning_rate=1e-3),
    ]
    results = run_sweep(configs, results_path=tmp_path / "results.csv")

    assert len(results) == 2
    assert (tmp_path / "results.csv").exists()
    for column in ("accuracy_before", "accuracy_after", "improvement", "final_loss"):
        assert column in results.columns
        assert results[column].notna().all()


@requires_deepfashion
def test_run_sweep_resumes_without_recomputing_completed_configs(tmp_path):
    """Simulates an interrupted-and-restarted sweep: run 2 configs, then
    call run_sweep again with those same 2 plus 1 new one, pointed at
    the same results file. The 2 already-completed configs should come
    back bit-identical to their first run (reused, not retrained,
    retraining would almost certainly produce different values, since
    model init and data order are both randomized), and only the new
    config should actually run.
    """
    results_path = tmp_path / "results.csv"
    configs = [
        TrainingConfig(num_steps=2, num_categories=10),
        TrainingConfig(num_steps=2, num_categories=10, learning_rate=1e-3),
    ]
    first_run = run_sweep(configs, results_path=results_path)

    resumed_configs = configs + [TrainingConfig(num_steps=2, num_categories=10, margin=0.4)]
    second_run = run_sweep(resumed_configs, results_path=results_path)

    assert len(second_run) == 3
    assert second_run.iloc[0]["accuracy_after"] == first_run.iloc[0]["accuracy_after"]
    assert second_run.iloc[1]["accuracy_after"] == first_run.iloc[1]["accuracy_after"]


@requires_deepfashion
def test_run_staged_search_carries_winner_forward(tmp_path):
    """Two tiny stages, each with 2 options, confirms: both stages ran,
    the summary records one winner per stage, and the final config
    actually reflects stage 2's winner (not stage 1's, not the base
    default), proving the carry-forward mechanism works, not just that
    both stages execute independently.
    """
    base = TrainingConfig(num_steps=2, num_categories=10)

    best_config, summary = run_staged_search(
        stages=[
            {"freeze_until_block": [6, 8]},
            {"margin": [0.1, 0.3]},
        ],
        base_config=base,
        results_dir=tmp_path,
    )

    assert len(summary) == 2
    assert set(summary["stage"]) == {1, 2}
    assert best_config.freeze_until_block in (6, 8)
    assert best_config.margin in (0.1, 0.3)
    # Regression check: carrying a winner forward once went through a
    # pandas row (a Series, single dtype), which silently upcast ints to
    # floats and crashed a later list slice. Pinning the type here, not
    # just the value, so that bug can't quietly come back.
    assert type(best_config.freeze_until_block) is int
    # Fields untouched by any stage should still match the base config
    assert best_config.num_steps == base.num_steps
    assert best_config.num_categories == base.num_categories

    assert (tmp_path / "staged_search_stage_1.csv").exists()
    assert (tmp_path / "staged_search_stage_2.csv").exists()
    assert (tmp_path / "staged_search_summary.csv").exists()
