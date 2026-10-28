"""Smoke test confirming Marqo-FashionCLIP's pretrained weights load and
run end-to-end on one real sample image. Downloads the pretrained weights
from Hugging Face on first run (cached locally afterward); needs network
access the first time it's run.
"""

import open_clip
import pytest
import torch

from stylematch.config import DEEPFASHION_DIR
from stylematch.similarity.dataset import DeepFashionCAPDataset

MODEL_ID = "hf-hub:Marqo/marqo-fashionCLIP"

pytestmark = pytest.mark.skipif(
    not (DEEPFASHION_DIR / "list_category_img.txt").exists(),
    reason="DeepFashion CAP subset not downloaded locally",
)


def test_backbone_loads_and_runs_on_one_image():
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval()

    # Reuses our own Dataset, but with the backbone's own required
    # preprocessing passed in as `transform`, instead of the mock
    # resize-and-divide-by-255 placeholder dataset.py falls back to.
    dataset = DeepFashionCAPDataset(split="train", sample_size=1, transform=preprocess_val)
    image, _category_label, _attribute_label = dataset[0]
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_batch)

    assert embedding.shape[0] == 1
    assert embedding.ndim == 2


if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval()
    print("Loaded.")

    dataset = DeepFashionCAPDataset(split="train", sample_size=1, transform=preprocess_val)
    image, category_label, _attribute_label = dataset[0]
    print(f"Sample image tensor shape (after backbone's own preprocessing): {tuple(image.shape)}")
    print(f"Sample image category label: {category_label}")

    image_batch = image.unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_batch)
    print(f"Embedding shape: {tuple(embedding.shape)}")
