"""Hyperparameter sweep for the similarity engine: runs training +
before/after validation-accuracy evaluation across multiple configs,
so they can be compared side by side. Validation split only, never
the test set, see evaluate.py."""

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
from pathlib import Path

from loguru import logger
import pandas as pd

from stylematch.config import REPORTS_DIR
from stylematch.similarity.dataset import get_qualifying_categories
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.train import run_training_loop


@dataclass
class TrainingConfig:
    """One set of tunable training hyperparameters."""

    learning_rate: float = 1e-5
    freeze_until_block: int = 8
    margin: float = 0.2
    categories_per_batch: int = 4
    images_per_category: int = 4
    num_categories: int | None = 20
    num_steps: int = 30
    embedding_dim: int = 256
    max_images_per_category: int | None = None
    weight_by_category_size: bool = False
    category_inclusion_threshold: int | None = None


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
    """Finds a row in `existing_results` whose hyperparameters match
    `config` exactly, for resuming an interrupted sweep. Floats get a
    tolerance (CSV round-tripping loses precision); None fields are
    matched via pd.isna since CSV round-trips them as NaN, not None."""
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
    configs: list[TrainingConfig], results_path: Path | None = REPORTS_DIR / "sweep_results.csv"
) -> pd.DataFrame:
    """Runs training + before/after validation retrieval accuracy for
    each config. Resumable: a config already in `results_path` is
    reused instead of retrained, so a timed-out Slurm job only redoes
    remaining work on resubmit.

    Args:
        configs: the hyperparameter combinations to try.
        results_path: CSV saved after every config (so a crash loses
            nothing) and read on resume. None disables both.

    Returns:
        One row per config: hyperparameters, before/after accuracy,
        improvement, final training loss.
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

        # Computed once per config, shared across before-eval, training,
        # and after-eval, so all three use the same category set.
        inclusion_threshold = (
            config.images_per_category
            if config.category_inclusion_threshold is None
            else config.category_inclusion_threshold
        )
        allowed_categories = get_qualifying_categories(inclusion_threshold, split="train")
        if config.num_categories is not None:
            allowed_categories = set(sorted(allowed_categories)[: config.num_categories])

        baseline_model = SimilarityModel(
            embedding_dim=config.embedding_dim, freeze_until_block=config.freeze_until_block
        )
        before = evaluate_retrieval_accuracy(
            baseline_model,
            split="val",
            min_images_per_category=config.images_per_category,
            allowed_categories=allowed_categories,
        )

        trained_model, losses = run_training_loop(
            num_steps=config.num_steps,
            categories_per_batch=config.categories_per_batch,
            images_per_category=config.images_per_category,
            learning_rate=config.learning_rate,
            freeze_until_block=config.freeze_until_block,
            embedding_dim=config.embedding_dim,
            margin=config.margin,
            max_images_per_category=config.max_images_per_category,
            weight_by_category_size=config.weight_by_category_size,
            allowed_categories=allowed_categories,
        )
        after = evaluate_retrieval_accuracy(
            trained_model,
            split="val",
            min_images_per_category=config.images_per_category,
            allowed_categories=allowed_categories,
        )

        row = {
            **asdict(config),
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
            "improvement": after["accuracy"] - before["accuracy"],
            "random_baseline": before["random_baseline"],
            "final_loss": losses[-1],
        }
        rows.append(row)
        logger.success(
            f"Config {i}: before={before['accuracy']:.3f} after={after['accuracy']:.3f} "
            f"improvement={row['improvement']:+.3f}"
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
) -> tuple[TrainingConfig, pd.DataFrame]:
    """Runs a sequence of small sweeps: each stage varies only its own
    named parameter(s), holding everything else at the previous
    stage's winner, then carries that winner forward automatically.

    Args:
        stages: ordered dicts naming which field(s) to vary and what
            values to try, e.g. [{"margin": [0.1, 0.2, 0.3]}, ...].
        base_config: starting point for fields no stage has touched yet.
        results_dir: each stage's results + a final summary CSV are
            saved here, so the run's reasoning stays inspectable.

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
            configs, results_path=results_dir / f"staged_search_stage_{stage_num}.csv"
        )

        winner_idx = stage_results["improvement"].idxmax()
        winning_config = configs[winner_idx]
        winning_improvement = stage_results.loc[winner_idx, "improvement"]

        for name in param_names:
            current_best = replace(current_best, **{name: getattr(winning_config, name)})

        winning_values = {name: getattr(winning_config, name) for name in param_names}
        summary_rows.append(
            {
                "stage": stage_num,
                "varied": ", ".join(param_names),
                **winning_values,
                "improvement": winning_improvement,
            }
        )
        logger.success(
            f"Stage {stage_num} winner: {winning_values} (improvement={winning_improvement:+.3f})"
        )

    summary = pd.DataFrame(summary_rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "staged_search_summary.csv", index=False)
    logger.success(f"Staged search done. Final best config: {current_best}")

    return current_best, summary


def run_staged_search_lite(
    stages: list[dict[str, list]] | None = None,
    base_config: TrainingConfig | None = None,
    results_dir: Path = REPORTS_DIR / "sweep_lite",
) -> tuple[TrainingConfig, pd.DataFrame]:
    """Same mechanics as run_staged_search, scaled down for a fast local
    check (small grids, few steps/categories) -- confirms the search
    machinery works, not meant for real hyperparameter conclusions.
    Writes to its own results_dir so it never mixes with real results.

    Args:
        stages: same shape as run_staged_search's `stages`; defaults to
            a small 2-stage version.
        base_config: defaults to a fast local config.
        results_dir: defaults to a dedicated "lite" subdirectory.
    """
    if stages is None:
        stages = [
            {"learning_rate": [1e-4, 1e-5], "freeze_until_block": [6, 8]},
            {"margin": [0.1, 0.2]},
        ]
    if base_config is None:
        base_config = TrainingConfig(num_steps=5, num_categories=10)

    return run_staged_search(stages=stages, base_config=base_config, results_dir=results_dir)


if __name__ == "__main__":
    # learning_rate and freeze_until_block are searched jointly (both
    # govern how much the pretrained weights move, so they're coupled);
    # margin and num_steps are weaker-coupled, later, separate stages.
    best_config, summary = run_staged_search(
        stages=[
            {
                "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
                "freeze_until_block": [2, 4, 6, 8, 10, 12],
            },
            {"margin": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]},
            {"num_steps": [30, 60, 100, 200, 500]},
        ],
        base_config=TrainingConfig(num_categories=None),
    )
    print(summary)
    print("Final best config:", best_config)
