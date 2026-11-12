import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Overridable via a DEEPFASHION_DIR env var (or a .env file, already
# loaded above) so the same repo can point at a different images
# location per machine without any code change: locally, the default
# below (repo-relative) is fine; on Great Lakes, the real dataset lives
# on /scratch instead, set via .env there, never committed (.env is
# gitignored).
DEEPFASHION_DIR = Path(os.environ.get("DEEPFASHION_DIR", RAW_DATA_DIR / "deepfashion"))

# Same override pattern as DEEPFASHION_DIR above, for the same reason:
# checkpoints/results currently land wherever the repo itself is cloned
# (fine by default, e.g. Great Lakes' /home persists reliably), but
# either can be redirected elsewhere (e.g. if checkpoints alone ever
# need to move off /home's smaller quota) without touching code.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJ_ROOT / "models"))

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", PROJ_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
