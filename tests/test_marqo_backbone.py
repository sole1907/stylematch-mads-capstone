"""Smoke test confirming Marqo-FashionCLIP's pretrained weights load and
run end-to-end on one real sample image. Downloads the pretrained weights
from Hugging Face on first run (cached locally afterward); needs network
access the first time it's run.

Loading approach (open_clip, hf-hub: prefix) confirmed against the
model's own card: https://huggingface.co/Marqo/marqo-fashionCLIP
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


def load_and_embed_one_sample():
    """Loads the real backbone and runs one real DeepFashion image through
    it. Shared by the test and the __main__ block below, so the two don't
    drift out of sync with each other."""
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval()

    # Reuses our own Dataset, but with the backbone's own required
    # preprocessing passed in as `transform`, instead of the mock
    # resize-and-divide-by-255 placeholder dataset.py falls back to.
    dataset = DeepFashionCAPDataset(split="train", sample_size=1, transform=preprocess_val)
    image, category_label, _attribute_label = dataset[0]
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_batch)

    return image, category_label, embedding


def test_backbone_loads_and_runs_on_one_image():
    _image, _category_label, embedding = load_and_embed_one_sample()
    assert embedding.shape[0] == 1
    assert embedding.ndim == 2


if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    image, category_label, embedding = load_and_embed_one_sample()
    print(f"Sample image tensor shape (after backbone's own preprocessing): {tuple(image.shape)}")
    print(f"Sample image category label: {category_label}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
