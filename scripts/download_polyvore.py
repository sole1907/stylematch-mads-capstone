"""Download the Polyvore Outfits dataset (nondisjoint split).

Gated dataset on Hugging Face, requires requesting access first:
https://huggingface.co/datasets/mvasil/polyvore-outfits
Once approved, authenticate locally (`hf auth login`, or set an
HF_TOKEN env var) before running this script.

See references/polyvore_outfits_schema.md for what's inside these files.

Two steps, not one, since images aren't loose files in this repo like
DeepFashion's were -- they're embedded inside parquet shards:
  1. `download`: pulls the real annotation files (train/valid/test.json,
     compatibility_*.txt, fill_in_blank_*.json, polyvore_item_metadata.json)
     plus the raw parquet shards holding the images.
  2. `extract-images`: reads those shards and writes out
     images/<item_id>.jpg, reconstructing the loose-file layout our
     dataset interface expects.

One-off setup utility, not imported by any training code. Run directly,
or via `make data-compatibility`.
"""

from huggingface_hub import snapshot_download
from loguru import logger
import pyarrow.parquet as pq
import typer

from stylematch.config import POLYVORE_DIR

app = typer.Typer()

REPO_ID = "mvasil/polyvore-outfits"

ALLOW_PATTERNS = [
    "polyvore_item_metadata.json",
    "nondisjoint/train.json",
    "nondisjoint/valid.json",
    "nondisjoint/test.json",
    "nondisjoint/compatibility_*.txt",
    "nondisjoint/fill_in_blank_*.json",
    "data/nondisjoint/*.parquet",
]


@app.command()
def download():
    """Downloads the raw annotation files + parquet image shards into
    data/raw/polyvore_outfits/. Requires HF access already granted and
    local authentication.
    """
    POLYVORE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {REPO_ID} (nondisjoint) into {POLYVORE_DIR}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=POLYVORE_DIR,
    )
    logger.success("Done. Run `extract-images` next to reconstruct images/<item_id>.jpg.")


@app.command()
def extract_images(batch_size: int = 256):
    """Reads the downloaded parquet shards and writes out
    images/<item_id>.jpg, one file per unique item.

    Reads each shard in row-group batches via pyarrow rather than
    loading it whole into one DataFrame -- a single shard's image
    bytes can be large enough (over 1GB on disk for this dataset) to
    get OOM-killed on a memory-capped node, e.g. an HPC login node
    shared across many interactive users.

    Prints the first shard's actual columns before attempting
    extraction. If ITEM_ID_COLUMN/IMAGE_COLUMN below don't match what's
    printed, update them and re-run rather than trusting a blind guess.
    """
    parquet_dir = POLYVORE_DIR / "data" / "nondisjoint"
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        logger.error(f"No parquet files found in {parquet_dir}. Run `download` first.")
        raise typer.Exit(1)

    first_columns = pq.ParquetFile(shards[0]).schema_arrow.names
    logger.info(f"First shard's columns: {first_columns}")

    item_id_column = "item_id" if "item_id" in first_columns else None
    image_column = "image" if "image" in first_columns else None
    if item_id_column is None or image_column is None:
        logger.error(
            "Expected columns 'item_id' and 'image' weren't both found above. "
            "Update ITEM_ID_COLUMN/IMAGE_COLUMN in this script to match the "
            "real column names, then re-run."
        )
        raise typer.Exit(1)

    images_dir = POLYVORE_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for shard in shards:
        shard_written = 0
        parquet_file = pq.ParquetFile(shard)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=[item_id_column, image_column]
        ):
            item_ids = batch.column(item_id_column).to_pylist()
            images = batch.column(image_column).to_pylist()
            for item_id, image_bytes in zip(item_ids, images):
                out_path = images_dir / f"{item_id}.jpg"
                if out_path.exists():
                    skipped += 1
                    continue
                # HF image columns are commonly a dict ({"bytes": ..., "path":
                # ...}) rather than raw bytes directly -- handle both.
                if isinstance(image_bytes, dict):
                    image_bytes = image_bytes.get("bytes")
                out_path.write_bytes(image_bytes)
                written += 1
                shard_written += 1
        logger.info(f"{shard.name}: {shard_written} images written")

    logger.success(f"Done. {written} images written, {skipped} already existed.")


if __name__ == "__main__":
    app()
