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

# Overridable via env var/.env so the same repo can point at a
# different images location per machine (e.g. Great Lakes' /scratch).
DEEPFASHION_DIR = Path(os.environ.get("DEEPFASHION_DIR", RAW_DATA_DIR / "deepfashion"))

# Same override pattern, for the compatibility engine's dataset. See
# references/polyvore_outfits_schema.md for what's inside.
POLYVORE_DIR = Path(os.environ.get("POLYVORE_DIR", RAW_DATA_DIR / "polyvore_outfits"))

# Same override pattern, for the stretch-goal African-fashion
# generalization check. Points at the actual images folder, inside
# the zip's own nested folder.
AFRIFASHION_DIR = Path(
    os.environ.get("AFRIFASHION_DIR", RAW_DATA_DIR / "afrifashion1600" / "AFRIFASHION1600")
)

# Same override pattern, in case checkpoints need to move off a
# smaller quota (e.g. Great Lakes' /home) without touching code.
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
