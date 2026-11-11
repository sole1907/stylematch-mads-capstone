"""Runs a fast, small-scale hyperparameter search locally, to sanity
check the sweep machinery (joint-stage winner carry-forward, resume-on-
rerun) works end to end, see sweep.py's run_staged_search_lite. Not
meant to produce trustworthy hyperparameter conclusions, that's what
the real, full-scale search (sweep.py's __main__, sized for Great
Lakes) is for.

Run directly:
    python scripts/run_similarity_sweep_lite.py
"""

import typer

from stylematch.similarity.sweep import run_staged_search_lite

app = typer.Typer()


@app.command()
def main() -> None:
    best_config, summary = run_staged_search_lite()
    print(summary)
    print("Final best config:", best_config)


if __name__ == "__main__":
    app()
