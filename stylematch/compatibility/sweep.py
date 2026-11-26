"""Hyperparameter sweep for the compatibility engine: runs training +
before/after FITB accuracy + AUC evaluation across multiple configs.
Validation split only, never test.

AUC drives config selection, not FITB accuracy: it's a continuous
score over 10,000 outfits, more stable than FITB's coarser 1-of-4
accuracy over 5,000 questions. FITB is still recorded, just not
what picks the winner."""

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
from pathlib import Path

from loguru import logger
import pandas as pd

from stylematch.compatibility.evaluate import evaluate_compatibility
from stylematch.compatibility.model import CompatibilityModel
from stylematch.compatibility.train import run_training_loop
from stylematch.config import REPORTS_DIR


@dataclass
class TrainingConfig:
    """One set of tunable training hyperparameters."""

    learning_rate: float = 1e-5
    freeze_until_block: int = 8
    margin: float = 0.2
    outfits_per_batch: int = 8
    num_steps: int = 30
    embedding_dim: int = 256


def build_grid(
    learning_rate: list[float] = (1e-5,),
    num_steps: list[int] = (30,),
    margin: list[float] = (0.2,),
    freeze_until_block: list[int] = (8,),
    **fixed_kwargs,
) -> list[TrainingConfig]:
    """Cartesian product of learning_rate / num_steps / margin /
    freeze_until_block as TrainingConfig objects. Other fields can be
    fixed across the grid via fixed_kwargs."""
    combos = product(learning_rate, num_steps, margin, freeze_until_block)
    return [
        TrainingConfig(
            learning_rate=lr,
            num_steps=steps,
            margin=m,
            freeze_until_block=freeze,
            **fixed_kwargs,
        )
        for lr, steps, m, freeze in combos
    ]


def _find_matching_row(config: TrainingConfig, existing_results: pd.DataFrame) -> dict | None:
    """Compares against each field's actual type on the live config
    object, floats get a tolerance, ints/others need exact equality.
    Returns the matching row as a dict, or None if no match exists.
    """
    config_values = asdict(config)
    for _, row in existing_results.iterrows():
        if all(
            pd.isna(row[name])
            if value is None
            else math.isclose(value, float(row[name]), rel_tol=1e-9, abs_tol=1e-12)
            if isinstance(value, float)
            else row[name] == value
            for name, value in config_values.items()
        ):
            return row.to_dict()
    return None


def run_sweep(
    configs: list[TrainingConfig],
    results_path: Path | None = REPORTS_DIR / "compatibility_sweep_results.csv",
    max_fitb_questions: int | None = None,
    max_compat_outfits: int | None = None,
) -> pd.DataFrame:
    """Runs training + before/after FITB accuracy + AUC for each
    config. Resumable: a config already in `results_path` is reused
    instead of retrained.

    Args:
        configs: the hyperparameter combinations to try.
        results_path: CSV saved after every config. None disables
            saving and resume.
        max_fitb_questions / max_compat_outfits: forwarded to
            evaluate_compatibility, for a fast local check.

    Returns:
        One row per config: hyperparameters, before/after FITB + AUC,
        both improvements, final loss. auc_improvement picks winners.
    """
    existing_results = None
    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            existing_results = pd.read_csv(results_path)
            logger.info(
                f"Found existing results at {results_path} ({len(existing_results)} rows), "
                f"will skip any configs already completed there"
            )

    rows = []
    skipped = 0
    for i, config in enumerate(configs, start=1):
        if existing_results is not None:
            match = _find_matching_row(config, existing_results)
            if match is not None:
                logger.info(f"Config {i}/{len(configs)}: already completed, reusing saved result")
                rows.append(match)
                skipped += 1
                continue

        logger.info(f"=== Config {i}/{len(configs)}: {config} ===")

        baseline_model = CompatibilityModel(
            embedding_dim=config.embedding_dim, freeze_until_block=config.freeze_until_block
        )
        before = evaluate_compatibility(
            baseline_model,
            split="valid",
            max_fitb_questions=max_fitb_questions,
            max_compat_outfits=max_compat_outfits,
        )

        trained_model, losses = run_training_loop(
            num_steps=config.num_steps,
            outfits_per_batch=config.outfits_per_batch,
            learning_rate=config.learning_rate,
            freeze_until_block=config.freeze_until_block,
            embedding_dim=config.embedding_dim,
            margin=config.margin,
        )
        after = evaluate_compatibility(
            trained_model,
            split="valid",
            max_fitb_questions=max_fitb_questions,
            max_compat_outfits=max_compat_outfits,
        )

        row = {
            **asdict(config),
            "fitb_accuracy_before": before["fitb_accuracy"],
            "fitb_accuracy_after": after["fitb_accuracy"],
            "fitb_improvement": after["fitb_accuracy"] - before["fitb_accuracy"],
            "auc_before": before["compatibility_auc"],
            "auc_after": after["compatibility_auc"],
            "auc_improvement": after["compatibility_auc"] - before["compatibility_auc"],
            "final_loss": losses[-1],
        }
        rows.append(row)
        logger.success(
            f"Config {i}: AUC before={before['compatibility_auc']:.3f} "
            f"after={after['compatibility_auc']:.3f} improvement={row['auc_improvement']:+.3f} "
            f"| FITB before={before['fitb_accuracy']:.3f} after={after['fitb_accuracy']:.3f}"
        )

        if results_path is not None:
            # Rewritten after every config, not appended, so the file
            # is always a valid, complete CSV if interrupted.
            pd.DataFrame(rows).to_csv(results_path, index=False)

    results = pd.DataFrame(rows)
    if results_path is not None:
        logger.info(
            f"Saved sweep results to {results_path}"
            + (f" ({skipped}/{len(configs)} resumed from a previous run)" if skipped else "")
        )

    return results


def run_staged_search(
    stages: list[dict[str, list]],
    base_config: TrainingConfig = TrainingConfig(),
    results_dir: Path = REPORTS_DIR,
    max_fitb_questions: int | None = None,
    max_compat_outfits: int | None = None,
) -> tuple[TrainingConfig, pd.DataFrame]:
    """Runs a sequence of small sweeps: each stage varies only its own
    named parameter(s), holding everything else at the previous
    stage's winner (by auc_improvement, see module docstring), then
    carries that winner forward automatically.

    Args:
        stages: ordered dicts naming which field(s) to vary and what
            values to try.
        base_config: starting point for fields no stage has touched yet.
        results_dir: each stage's results + a final summary CSV are
            saved here.
        max_fitb_questions / max_compat_outfits: forwarded to
            run_sweep, for a fast local check.

    Returns:
        (best_config, summary) -- final TrainingConfig and a DataFrame
        of each stage's winner and improvement score.
    """
    current_best = base_config
    summary_rows = []

    for stage_num, stage_params in enumerate(stages, start=1):
        param_names = list(stage_params)
        logger.info(f"=== Stage {stage_num}/{len(stages)}: varying {param_names} ===")

        configs = [
            replace(current_best, **dict(zip(param_names, values)))
            for values in product(*stage_params.values())
        ]
        stage_results = run_sweep(
            configs,
            results_path=results_dir / f"compatibility_staged_search_stage_{stage_num}.csv",
            max_fitb_questions=max_fitb_questions,
            max_compat_outfits=max_compat_outfits,
        )

        winner_idx = stage_results["auc_improvement"].idxmax()
        winning_config = configs[winner_idx]
        winning_improvement = stage_results.loc[winner_idx, "auc_improvement"]

        for name in param_names:
            current_best = replace(current_best, **{name: getattr(winning_config, name)})

        winning_values = {name: getattr(winning_config, name) for name in param_names}
        summary_rows.append(
            {
                "stage": stage_num,
                "varied": ", ".join(param_names),
                **winning_values,
                "auc_improvement": winning_improvement,
            }
        )
        logger.success(
            f"Stage {stage_num} winner: {winning_values} (auc_improvement={winning_improvement:+.3f})"
        )

    summary = pd.DataFrame(summary_rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "compatibility_staged_search_summary.csv", index=False)
    logger.success(f"Staged search done. Final best config: {current_best}")

    return current_best, summary


def run_staged_search_lite(
    stages: list[dict[str, list]] | None = None,
    base_config: TrainingConfig | None = None,
    results_dir: Path = REPORTS_DIR / "compatibility_sweep_lite",
) -> tuple[TrainingConfig, pd.DataFrame]:
    """Same mechanics as run_staged_search, scaled down for a fast
    local check (small grids, few steps, capped eval subset) -- confirms
    the search machinery works, not meant for real conclusions. Writes
    to its own results_dir so it never mixes with real results.
    """
    if stages is None:
        stages = [
            {"learning_rate": [1e-4, 1e-5], "freeze_until_block": [6, 8]},
            {"margin": [0.1, 0.2]},
        ]
    if base_config is None:
        base_config = TrainingConfig(num_steps=5, outfits_per_batch=4)

    return run_staged_search(
        stages=stages,
        base_config=base_config,
        results_dir=results_dir,
        max_fitb_questions=20,
        max_compat_outfits=20,
    )


if __name__ == "__main__":
    # margin's values are scaled to a real check: an untrained model's
    # score_outfit() on a 40-outfit sample clustered real and fake
    # outfits both in roughly a 0.45-0.73 band, a ~0.27 spread.
    best_config, summary = run_staged_search(
        stages=[
            {
                "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
                "freeze_until_block": [2, 4, 6, 8, 10, 12],
            },
            {"margin": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
            {"num_steps": [30, 60, 100, 200, 500]},
        ],
        base_config=TrainingConfig(),
    )
    print(summary)
    print("Final best config:", best_config)
