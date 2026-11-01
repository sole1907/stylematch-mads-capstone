"""Hyperparameter sweep for the similarity engine: runs training +
before/after validation-accuracy evaluation across multiple configs, so
they can be compared side by side instead of tested one at a time by
hand. Validation split only, never the test set, see evaluate.py.
"""

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
from pathlib import Path

from loguru import logger
import pandas as pd

from stylematch.config import REPORTS_DIR
from stylematch.similarity.evaluate import evaluate_retrieval_accuracy
from stylematch.similarity.model import SimilarityModel
from stylematch.similarity.train import run_training_loop


@dataclass
class TrainingConfig:
    """One set of tunable training hyperparameters. Defaults match
    train.py's own defaults, except learning_rate, which defaults here
    to the value a real before/after comparison showed actually helps
    (1e-3, train.py's original default, was confirmed to actively hurt
    performance by over-disrupting the pretrained backbone)."""

    learning_rate: float = 1e-5
    freeze_until_block: int = 8
    margin: float = 0.2
    categories_per_batch: int = 4
    images_per_category: int = 4
    num_categories: int = 20
    num_steps: int = 30
    embedding_dim: int = 256


def build_grid(
    learning_rate: list[float] = (1e-5,),
    num_steps: list[int] = (30,),
    margin: list[float] = (0.2,),
    freeze_until_block: list[int] = (8,),
    **fixed_kwargs,
) -> list[TrainingConfig]:
    """Builds every combination (Cartesian product) of the given
    learning_rate / num_steps / margin / freeze_until_block values as
    TrainingConfig objects. Any other TrainingConfig field (e.g.
    num_categories) can be fixed across the whole grid via
    fixed_kwargs. Intended for unattended runs on real compute (Great
    Lakes), where you want to kick off the full grid once rather than
    build the combination list by hand.
    """
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
    """Looks for a row in `existing_results` whose hyperparameters match
    `config` exactly, used to detect already-completed configs when
    resuming an interrupted sweep. Compares against each field's actual
    type on the live config object, not whatever dtype the CSV happened
    to load as, floats get a tolerance (CSV round-tripping can lose a
    little precision), ints/others need exact equality. Returns the
    matching row as a dict, or None if no match exists.
    """
    config_values = asdict(config)
    for _, row in existing_results.iterrows():
        if all(
            math.isclose(value, float(row[name]), rel_tol=1e-9, abs_tol=1e-12)
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
    each config in turn.

    Resumable: if `results_path` already has results from a previous,
    interrupted run, any config that already has a matching saved result
    is reused instead of retrained, so restarting after a timeout (e.g.
    a Slurm job hitting its time limit) only does the remaining work,
    not the whole sweep over again.

    Args:
        configs: the hyperparameter combinations to try.
        results_path: where to save the comparison table as CSV,
            updated after *every* config completes, not just at the
            end, so a crash or timeout partway through a long
            unattended run doesn't lose results already finished. Also
            where completed-config detection reads from, on resume.
            Pass None to skip saving, in-memory DataFrame only, and to
            disable resume (nothing to resume from without a file).

    Returns:
        One row per config: its hyperparameters, accuracy before and
        after training, the improvement, and the final training loss.
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

        baseline_model = SimilarityModel(
            embedding_dim=config.embedding_dim, freeze_until_block=config.freeze_until_block
        )
        before = evaluate_retrieval_accuracy(
            baseline_model, split="val", num_categories=config.num_categories
        )

        trained_model, losses = run_training_loop(
            num_steps=config.num_steps,
            categories_per_batch=config.categories_per_batch,
            images_per_category=config.images_per_category,
            num_categories=config.num_categories,
            learning_rate=config.learning_rate,
            freeze_until_block=config.freeze_until_block,
            embedding_dim=config.embedding_dim,
            margin=config.margin,
        )
        after = evaluate_retrieval_accuracy(
            trained_model, split="val", num_categories=config.num_categories
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
            # Rewritten after every config, not appended, so the file is
            # always a valid, complete CSV if a long run gets interrupted,
            # never a half-written row.
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
    """Runs a sequence of small sweeps automatically: each stage varies
    only the parameter(s) named in that stage's dict, holding everything
    else at the best config found by the *previous* stage, then that
    stage's winner (by `improvement`) gets carried forward automatically.

    This is what turns "run stage 1, look at the winner, manually set up
    stage 2 with it" into something you kick off once and walk away
    from, no step-by-step babysitting needed. Much cheaper than a full
    combinatorial grid across all parameters at once, each stage only
    varies 1-2 fields, not the full cross product of everything.

    Args:
        stages: ordered list of dicts, each naming which TrainingConfig
            field(s) to vary that stage and what values to try, e.g.
            [{"freeze_until_block": [4, 6, 8, 10]},
             {"margin": [0.1, 0.2, 0.3]},
             {"num_steps": [30, 60, 100]}].
        base_config: starting point for every field not yet touched by
            an earlier stage.
        results_dir: each stage's full results are saved as their own
            CSV (staged_search_stage_1.csv, ...), plus a final
            staged_search_summary.csv recording the winner chosen at
            each stage, so the whole run's reasoning is inspectable
            afterward, not just the final config.

    Returns:
        (best_config, summary): the final TrainingConfig after all
        stages' winners are applied, and a DataFrame of what was chosen
        at each stage and why (its improvement score).
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

        # Pulling the winning *config object* directly (not reading its
        # values back out of the DataFrame row): stage_results.loc[idx]
        # would return that row as a pandas Series, and a Series can
        # only hold one dtype, so mixing this row's int and float
        # columns silently upcasts everything in it to float (confirmed
        # this bug directly: freeze_until_block=6 became 6.0, which then
        # crashed a later list slice expecting an int). Going back to
        # the original configs list avoids that round-trip entirely.
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


if __name__ == "__main__":
    # learning_rate and freeze_until_block are searched *jointly* in one
    # stage, not sequentially: both govern how much the pretrained
    # weights get disrupted per step (freeze depth controls how many
    # parameters can move, LR controls how far each one moves), so
    # they're plausibly coupled, fixing one before searching the other
    # risks finding a freeze depth that's only good conditional on a
    # possibly-wrong LR. The 1e-4 default used elsewhere came from a
    # single small-data sweep and was never actually confirmed to hold
    # at a different dataset size, don't treat it as settled, this
    # stage re-checks it rather than assuming it.
    #
    # margin and num_steps are kept as separate, later stages, weaker
    # coupling to LR/freeze depth, sequential narrowing is a reasonable
    # tradeoff there to keep the total search tractable.
    #
    # learning_rate includes 1e-6 (not just 1e-3/1e-4/1e-5): an earlier
    # sweep at freeze_until_block=8 showed 1e-6 performing worse than
    # 1e-5, but freeze_until_block now ranges much wider here (2-12),
    # and more trainable parameters at the aggressive end (e.g. 2) could
    # plausibly need an even smaller LR than anything tested before to
    # stay stable. Not adding anything above 1e-3, every result so far
    # points toward higher being more destructive, not less.
    best_config, summary = run_staged_search(
        stages=[
            {
                "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
                "freeze_until_block": [2, 4, 6, 8, 10, 12],
            },
            {"margin": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]},
            {"num_steps": [30, 60, 100, 200, 500]},
        ],
        base_config=TrainingConfig(),
    )
    print(summary)
    print("Final best config:", best_config)
