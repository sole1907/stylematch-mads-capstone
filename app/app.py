"""StyleMatch demo app.

5 tabs, matching the planned build order:
1. Similarity: upload 1 image -> top-7 similar items. Real inference.
2. Compatibility (naive): upload 1 OR MORE garments (e.g. a shirt
   and pants already picked out) -> top-7 compatible items to add,
   no category filtering. score_outfit() takes any number >= 2 total
   items (context + candidate), so more context is just as valid as
   one -- and closer to the FITB task we actually measured accuracy
   on, which always used multiple context items, not isolated pairs.
   Real inference.
3. Compatibility (labeled): same, plus a category dropdown per
   garment (the user states each uploaded item's own category) to
   filter out candidates matching any of those categories. Real
   inference.
4. Compatibility (infer): same, but each garment's category is
   inferred directly in Polyvore's own 11-category vocabulary --
   similarity-engine neighbors against a Polyvore catalog, majority
   vote. No DeepFashion-to-Polyvore mapping step, and no blind spot
   for categories DeepFashion doesn't have (shoes, bags, jewellery,
   etc.). Exploratory -- this inference step isn't separately
   validated, unlike the labeled tab's user-stated category. Real
   inference.
5. Stretch goal: African garments. Category inferred by similarity
   against the AFRIFASHION catalog (its own 8-class vocabulary), then
   mapped to the closest Polyvore category via AFRIFASHION_TO_POLYVORE
   so "compatible" results can be drawn from the much richer Polyvore
   catalog instead of the 1,600-image African-only one. Real inference.

Requires models/similarity_final.pt, models/compatibility_final.pt,
models/similarity_catalog.pt, models/compatibility_catalog.pt,
models/polyvore_similarity_catalog.pt to already exist (see
scripts/train_*_final.py, scripts/build_*_catalog.py).

Run directly:
    python app/app.py
"""

from collections import Counter

from PIL import Image
import gradio as gr
import torch

from stylematch.catalog import AFRIFASHION_CLASSES, load_catalog, search_by_compatibility, search_by_similarity
from stylematch.compatibility.dataset import CATEGORY_TO_LABEL, SEMANTIC_CATEGORIES
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import AFRIFASHION_DIR, DEEPFASHION_DIR, MODELS_DIR, POLYVORE_DIR
from stylematch.similarity.model import SimilarityModel

TOP_K = 7
INFER_NEIGHBORS = 5
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10

# AFRIFASHION's 8 classes mapped to the closest real Polyvore semantic
# category, so the stretch tab's "compatible" search can use the much
# richer Polyvore catalog instead of the 1,600-image African-only one.
AFRIFASHION_TO_POLYVORE = {
    "African_Blouse": "tops",
    "African_Shirts": "tops",
    "Agbada": "all-body",
    "Buba and Trouser": "all-body",
    "Gele": "hats",
    "Gown": "all-body",
    "Skirt and Blouse": "all-body",
    "Wrapper and Blouse": "all-body",
}


similarity_model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
similarity_model.load_state_dict(
    torch.load(MODELS_DIR / "similarity_final.pt", map_location=similarity_model.device)
)
similarity_catalog = load_catalog(MODELS_DIR / "similarity_catalog.pt")

# Polyvore items embedded via the SIMILARITY engine (see
# scripts/build_polyvore_similarity_catalog.py) -- lets category
# inference search Polyvore's own native 11-category vocabulary
# directly, with no DeepFashion-to-Polyvore mapping step.
polyvore_similarity_catalog = load_catalog(MODELS_DIR / "polyvore_similarity_catalog.pt")
polyvore_similarity_id_to_category = dict(
    zip(polyvore_similarity_catalog["identifiers"], polyvore_similarity_catalog["category_labels"].tolist())
)

compatibility_model = CompatibilityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
compatibility_model.load_state_dict(
    torch.load(MODELS_DIR / "compatibility_final.pt", map_location=compatibility_model.device)
)
compatibility_catalog = load_catalog(MODELS_DIR / "compatibility_catalog.pt")

# Stretch goal: similarity search stays within AFRIFASHION's own
# 8-class label space (both catalogs built from AfrifashionDataset).
afrifashion_similarity_catalog = load_catalog(MODELS_DIR / "afrifashion_similarity_catalog.pt")
afrifashion_similarity_id_to_category = dict(
    zip(afrifashion_similarity_catalog["identifiers"], afrifashion_similarity_catalog["category_labels"].tolist())
)


def _filter_none(*images) -> list[Image.Image]:
    """Drops any unfilled optional upload box (None) from a fixed set
    of gr.Image inputs."""
    return [img for img in images if img is not None]


@torch.no_grad()
def find_similar(uploaded_image) -> list:
    if uploaded_image is None:
        return []
    tensor = similarity_model.preprocess_val(uploaded_image).unsqueeze(0)
    query_embedding = similarity_model(tensor)[0].cpu()

    results = search_by_similarity(query_embedding, similarity_catalog, top_k=TOP_K)
    return [
        (Image.open(DEEPFASHION_DIR / image_name).convert("RGB"), f"score: {score:.3f}")
        for image_name, score in results
    ]


@torch.no_grad()
def _find_compatible(images: list[Image.Image], exclude_category_labels=None) -> list:
    if not images:
        return []
    tensors = torch.stack([compatibility_model.preprocess_val(img) for img in images])
    context_embeddings = compatibility_model(tensors).cpu()

    results = search_by_compatibility(
        context_embeddings, compatibility_catalog, top_k=TOP_K, exclude_category_labels=exclude_category_labels
    )
    return [
        (Image.open(POLYVORE_DIR / "images" / f"{item_id}.jpg").convert("RGB"), f"score: {score:.3f}")
        for item_id, score in results
    ]


def find_compatible_naive(image1, image2, image3) -> list:
    return _find_compatible(_filter_none(image1, image2, image3))


def find_compatible_labeled(image1, category1, image2, category2, image3, category3) -> list:
    """Each garment gets its own category -- excludes candidates
    matching ANY of them, not just one shared category (a shirt +
    pants upload needs both "tops" and "bottoms" excluded, not just
    whichever one a single shared dropdown happened to hold)."""
    pairs = [(image1, category1), (image2, category2), (image3, category3)]
    images = _filter_none(*(img for img, _ in pairs))
    exclude = {CATEGORY_TO_LABEL[cat] for img, cat in pairs if img is not None and cat}
    return _find_compatible(images, exclude_category_labels=exclude or None)


@torch.no_grad()
def infer_category(image: Image.Image) -> str:
    """Embeds via the similarity engine, finds its top-5 nearest
    neighbors in the Polyvore similarity catalog, takes a majority
    vote of their category. Already in Polyvore's native vocabulary,
    so no mapping step -- and no blind spot for categories DeepFashion
    doesn't have (shoes, bags, jewellery, etc.)."""
    tensor = similarity_model.preprocess_val(image).unsqueeze(0)
    query_embedding = similarity_model(tensor)[0].cpu()
    neighbors = search_by_similarity(query_embedding, polyvore_similarity_catalog, top_k=INFER_NEIGHBORS)

    neighbor_labels = [polyvore_similarity_id_to_category[item_id] for item_id, _ in neighbors]
    majority_label = Counter(neighbor_labels).most_common(1)[0][0]
    return SEMANTIC_CATEGORIES[majority_label]


@torch.no_grad()
def find_compatible_infer(image1, image2, image3):
    images = _filter_none(image1, image2, image3)
    if not images:
        return [], ""

    inferred = [infer_category(img) for img in images]
    inferred_display = ", ".join(f"Garment {i + 1}: {cat}" for i, cat in enumerate(inferred))
    exclude = {CATEGORY_TO_LABEL[cat] for cat in inferred}
    results = _find_compatible(images, exclude_category_labels=exclude)
    return results, inferred_display


def _placeholder_results(image1, image2, image3) -> list:
    """Not wired to a model yet -- just echoes the upload(s) back so
    the gallery/layout can be checked end-to-end."""
    return _filter_none(image1, image2, image3)


@torch.no_grad()
def infer_afrifashion_category(image: Image.Image) -> str:
    """Same majority-vote mechanism as infer_category, against the
    AFRIFASHION similarity catalog -- returns one of AFRIFASHION_CLASSES,
    not a Polyvore category (see AFRIFASHION_TO_POLYVORE for that)."""
    tensor = similarity_model.preprocess_val(image).unsqueeze(0)
    query_embedding = similarity_model(tensor)[0].cpu()
    neighbors = search_by_similarity(query_embedding, afrifashion_similarity_catalog, top_k=INFER_NEIGHBORS)

    neighbor_labels = [afrifashion_similarity_id_to_category[name] for name, _ in neighbors]
    majority_label = Counter(neighbor_labels).most_common(1)[0][0]
    return AFRIFASHION_CLASSES[majority_label]


@torch.no_grad()
def find_similar_and_compatible_african(image):
    if image is None:
        return [], [], ""

    query_tensor = similarity_model.preprocess_val(image).unsqueeze(0)
    query_embedding = similarity_model(query_tensor)[0].cpu()
    similar_results = search_by_similarity(query_embedding, afrifashion_similarity_catalog, top_k=TOP_K)
    similar_gallery = [
        (Image.open(AFRIFASHION_DIR / name).convert("RGB"), f"score: {score:.3f}")
        for name, score in similar_results
    ]

    inferred = infer_afrifashion_category(image)
    polyvore_category = AFRIFASHION_TO_POLYVORE[inferred]
    exclude = {CATEGORY_TO_LABEL[polyvore_category]}

    # Compatible search runs against the main Polyvore catalog, not
    # the 1,600-image African-only one -- richer candidate pool, once
    # the inferred African category is mapped into Polyvore's space.
    compat_tensor = compatibility_model.preprocess_val(image).unsqueeze(0)
    compat_embedding = compatibility_model(compat_tensor).cpu()
    compat_results = search_by_compatibility(
        compat_embedding, compatibility_catalog, top_k=TOP_K, exclude_category_labels=exclude
    )
    compat_gallery = [
        (Image.open(POLYVORE_DIR / "images" / f"{item_id}.jpg").convert("RGB"), f"score: {score:.3f}")
        for item_id, score in compat_results
    ]

    inferred_display = f"Inferred category: {inferred} (Polyvore: {polyvore_category})"
    return similar_gallery, compat_gallery, inferred_display


with gr.Blocks(title="StyleMatch") as demo:
    gr.Markdown("# StyleMatch — fashion similarity & compatibility demo")

    with gr.Tab("Similarity"):
        gr.Markdown(f"Upload a garment, see its top-{TOP_K} most similar items.")
        similarity_input = gr.Image(type="pil", label="Upload a garment")
        similarity_button = gr.Button("Find similar items")
        similarity_output = gr.Gallery(label="Top matches", columns=TOP_K, object_fit="contain", allow_preview=False)
        similarity_button.click(fn=find_similar, inputs=[similarity_input], outputs=similarity_output)

    with gr.Tab("Compatibility (naive)"):
        gr.Markdown(
            f"Upload one, two, or three garments you already have (e.g. a shirt and pants), see "
            f"the top-{TOP_K} items to add. No category filtering."
        )
        with gr.Row():
            compat_naive_input1 = gr.Image(type="pil", label="Garment 1")
            compat_naive_input2 = gr.Image(type="pil", label="Garment 2 (optional)")
            compat_naive_input3 = gr.Image(type="pil", label="Garment 3 (optional)")
        compat_naive_button = gr.Button("Find compatible items")
        compat_naive_output = gr.Gallery(label="Top matches", columns=TOP_K, object_fit="contain", allow_preview=False)
        compat_naive_button.click(
            fn=find_compatible_naive,
            inputs=[compat_naive_input1, compat_naive_input2, compat_naive_input3],
            outputs=compat_naive_output,
        )

    with gr.Tab("Compatibility (labeled)"):
        gr.Markdown(
            f"Upload one, two, or three garments and tell us EACH one's own category, see the "
            f"top-{TOP_K} items to add, excluding items from any of those same categories."
        )
        with gr.Row():
            with gr.Column():
                compat_labeled_input1 = gr.Image(type="pil", label="Garment 1")
                compat_labeled_category1 = gr.Dropdown(choices=SEMANTIC_CATEGORIES, label="Garment 1's category")
            with gr.Column():
                compat_labeled_input2 = gr.Image(type="pil", label="Garment 2 (optional)")
                compat_labeled_category2 = gr.Dropdown(choices=SEMANTIC_CATEGORIES, label="Garment 2's category")
            with gr.Column():
                compat_labeled_input3 = gr.Image(type="pil", label="Garment 3 (optional)")
                compat_labeled_category3 = gr.Dropdown(choices=SEMANTIC_CATEGORIES, label="Garment 3's category")
        compat_labeled_button = gr.Button("Find compatible items")
        compat_labeled_output = gr.Gallery(label="Top matches", columns=TOP_K, object_fit="contain", allow_preview=False)
        compat_labeled_button.click(
            fn=find_compatible_labeled,
            inputs=[
                compat_labeled_input1,
                compat_labeled_category1,
                compat_labeled_input2,
                compat_labeled_category2,
                compat_labeled_input3,
                compat_labeled_category3,
            ],
            outputs=compat_labeled_output,
        )

    with gr.Tab("Compatibility (infer)"):
        gr.Markdown(
            f"Upload one, two, or three garments, category inferred automatically (via the "
            f"similarity engine's nearest neighbors), see the top-{TOP_K} items to add. "
            "Exploratory -- category inference isn't separately validated."
        )
        with gr.Row():
            compat_infer_input1 = gr.Image(type="pil", label="Garment 1")
            compat_infer_input2 = gr.Image(type="pil", label="Garment 2 (optional)")
            compat_infer_input3 = gr.Image(type="pil", label="Garment 3 (optional)")
        compat_infer_button = gr.Button("Find compatible items")
        compat_infer_category = gr.Textbox(label="Inferred category", interactive=False)
        compat_infer_output = gr.Gallery(label="Top matches", columns=TOP_K, object_fit="contain", allow_preview=False)
        compat_infer_button.click(
            fn=find_compatible_infer,
            inputs=[compat_infer_input1, compat_infer_input2, compat_infer_input3],
            outputs=[compat_infer_output, compat_infer_category],
        )

    with gr.Tab("Stretch: African garments"):
        gr.Markdown(
            f"Upload an African garment, see its top-{TOP_K} similar AND top-{TOP_K} compatible "
            "items (category inferred the same way as the Infer tab). Stretch goal, time-permitting."
        )
        stretch_input = gr.Image(type="pil", label="Upload a garment")
        stretch_button = gr.Button("Find similar & compatible items")
        stretch_category = gr.Textbox(label="Inferred category", interactive=False)
        stretch_similar_output = gr.Gallery(label="Similar items", columns=TOP_K, object_fit="contain", allow_preview=False)
        stretch_compat_output = gr.Gallery(label="Compatible items", columns=TOP_K, object_fit="contain", allow_preview=False)
        stretch_button.click(
            fn=find_similar_and_compatible_african,
            inputs=[stretch_input],
            outputs=[stretch_similar_output, stretch_compat_output, stretch_category],
        )


if __name__ == "__main__":
    demo.launch()
