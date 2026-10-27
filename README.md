# stylematch-mads-capstone

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Two-stage visual fashion recommendation system (similarity + compatibility engines), SIADS 699 capstone, Team StyleMatch

## Setup

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/sole1907/stylematch-mads-capstone.git
cd stylematch-mads-capstone
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

See [`DATA_ACCESS.md`](DATA_ACCESS.md) for dataset sources, access methods,
and license restrictions.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         stylematch and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── scripts            <- One-off setup/utility scripts, not imported by any
│   │                     pipeline or training code
│   └── download_deepfashion.py  <- Fetches the DeepFashion CAP subset
│
├── setup.cfg          <- Configuration file for flake8
│
└── stylematch   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes stylematch a Python module
    │
    ├── config.py               <- Shared: paths, env vars, configuration
    │
    ├── plots.py                <- Shared: generic plotting helpers
    │
    ├── features.py             <- Not yet in use (generic scaffold stub,
    │                              undecided whether needed for image-based
    │                              models, revisit later)
    │
    ├── similarity              <- Similarity engine (DeepFashion)
    │   ├── __init__.py
    │   ├── dataset.py          <- PyTorch Dataset over the downloaded CAP
    │   │                          subset (see scripts/download_deepfashion.py)
    │   ├── model.py            <- Backbone + embedding head
    │   ├── train.py            <- Training loop (triplet/contrastive loss)
    │   └── predict.py          <- Retrieval inference
    │
    └── compatibility           <- Compatibility engine (Polyvore Outfits)
        ├── __init__.py
        ├── dataset.py          <- Polyvore data loading (outfits, FITB)
        ├── model.py            <- Compatibility scoring architecture
        ├── train.py            <- Training loop
        └── predict.py          <- Compatibility inference
```

--------

