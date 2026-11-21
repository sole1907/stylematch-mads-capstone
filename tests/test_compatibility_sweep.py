"""Tests for the compatibility engine's hyperparameter sweep utility.
build_grid's tests are pure combinatorics, no dataset needed. The rest
actually train, so they require the Polyvore Outfits dataset to be
downloaded (skipped otherwise), and use a tiny config (few steps, a
small capped eval subset) to check the mechanics work, not to produce
a meaningful comparison -- that's what sweep.py's own __main__ block
is for.
"""

import pytest

from stylematch.compatibility.sweep import (
    TrainingConfig,
    build_grid,
    run_sweep,
    run_staged_search,
    run_staged_search_lite,
)
from stylematch.config import POLYVORE_DIR


def test_build_grid_produces_full_cartesian_product():
    configs = build_grid(
        learning_rate=[1e-3, 1e-4],
        num_steps=[2, 3],
        margin=[0.1],
        freeze_until_block=[6, 8, 10],
    )
    # 2 learning_rate x 2 num_steps x 1 margin x 3 freeze_until_block
    assert len(configs) == 2 * 2 * 1 * 3
    assert all(isinstance(c, TrainingConfig) for c in configs)
    combos = {(c.learning_rate, c.num_steps, c.margin, c.freeze_until_block) for c in configs}
    assert len(combos) == len(configs)


def test_build_grid_passes_through_fixed_kwargs():
    configs = build_grid(learning_rate=[1e-4], outfits_per_batch=4)
    assert all(c.outfits_per_batch == 4 for c in configs)


requires_polyvore = pytest.mark.skipif(
    not (POLYVORE_DIR / "nondisjoint" / "train.json").exists(),
    reason="Polyvore Outfits dataset not downloaded locally",
)

# Small eval subset for all tests here -- mechanics, not meaningful numbers.
EVAL_KWARGS = {"max_fitb_questions": 8, "max_compat_outfits": 8}


@requires_polyvore
def test_run_sweep_produces_one_row_per_config(tmp_path):
    configs = [
        TrainingConfig(num_steps=2, outfits_per_batch=4),
        TrainingConfig(num_steps=2, outfits_per_batch=4, learning_rate=1e-3),
    ]
    results = run_sweep(configs, results_path=tmp_path / "results.csv", **EVAL_KWARGS)

    assert len(results) == 2
    assert (tmp_path / "results.csv").exists()
    for column in (
        "fitb_accuracy_before",
        "fitb_accuracy_after",
        "fitb_improvement",
        "auc_before",
        "auc_after",
        "auc_improvement",
        "final_loss",
    ):
        assert column in results.columns
        assert results[column].notna().all()


@requires_polyvore
def test_run_sweep_resumes_without_recomputing_completed_configs(tmp_path):
    """Simulates an interrupted-and-restarted sweep: run 2 configs,
    then call run_sweep again with those same 2 plus 1 new one, same
    results file. The 2 already-completed configs should come back
    bit-identical (reused, not retrained -- retraining would almost
    certainly produce different values, model init and batch order
    are both randomized), only the new config should actually run.
    """
    results_path = tmp_path / "results.csv"
    configs = [
        TrainingConfig(num_steps=2, outfits_per_batch=4),
        TrainingConfig(num_steps=2, outfits_per_batch=4, learning_rate=1e-3),
    ]
    first_run = run_sweep(configs, results_path=results_path, **EVAL_KWARGS)

    resumed_configs = configs + [TrainingConfig(num_steps=2, outfits_per_batch=4, margin=0.4)]
    second_run = run_sweep(resumed_configs, results_path=results_path, **EVAL_KWARGS)

    assert len(second_run) == 3
    assert second_run.iloc[0]["auc_after"] == first_run.iloc[0]["auc_after"]
    assert second_run.iloc[1]["auc_after"] == first_run.iloc[1]["auc_after"]


@requires_polyvore
def test_run_staged_search_carries_winner_forward(tmp_path):
    """Two tiny stages, each with 2 options, confirms: both stages
    ran, the summary records one winner per stage (by auc_improvement),
    and the final config reflects stage 2's winner, not stage 1's or
    the base default -- proving the carry-forward mechanism works."""
    base = TrainingConfig(num_steps=2, outfits_per_batch=4)

    best_config, summary = run_staged_search(
        stages=[
            {"freeze_until_block": [6, 8]},
            {"margin": [0.1, 0.3]},
        ],
        base_config=base,
        results_dir=tmp_path,
        **EVAL_KWARGS,
    )

    assert len(summary) == 2
    assert set(summary["stage"]) == {1, 2}
    assert best_config.freeze_until_block in (6, 8)
    assert best_config.margin in (0.1, 0.3)
    # Regression check, same one the similarity engine's sweep guards
    # against: carrying a winner forward once went through a pandas
    # row (a Series, single dtype), which silently upcast ints to
    # floats. Pinning the type here, not just the value.
    assert type(best_config.freeze_until_block) is int
    assert best_config.num_steps == base.num_steps
    assert best_config.outfits_per_batch == base.outfits_per_batch

    assert (tmp_path / "compatibility_staged_search_stage_1.csv").exists()
    assert (tmp_path / "compatibility_staged_search_stage_2.csv").exists()
    assert (tmp_path / "compatibility_staged_search_summary.csv").exists()


@requires_polyvore
def test_run_staged_search_lite_uses_small_defaults_and_runs(tmp_path):
    """Confirms the convenience wrapper produces a fast, small run
    end to end using its own built-in defaults, writing to the given
    results_dir rather than the real search's default location."""
    best_config, summary = run_staged_search_lite(results_dir=tmp_path)

    assert len(summary) == 2  # default: 2 stages
    assert best_config.num_steps == 5  # default base_config's fast setting
    assert best_config.outfits_per_batch == 4
    assert (tmp_path / "compatibility_staged_search_stage_1.csv").exists()
    assert (tmp_path / "compatibility_staged_search_stage_2.csv").exists()
