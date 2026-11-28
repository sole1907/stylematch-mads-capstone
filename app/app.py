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
3. Compatibility (labeled): same, plus a category dropdown (the
   user states the NEW garment's category) to filter same-category
   candidates out before ranking. Real inference.
4. Compatibility (infer): same, but the category is inferred from
   the similarity engine's nearest neighbors instead of asked for.
   Not wired yet -- needs a DeepFashion-to-Polyvore category mapping.
5. Stretch goal: African garments, both engines, category inferred
   the same way as tab 4. Not wired yet, lowest priority.

Requires models/similarity_final.pt, models/compatibility_final.pt,
models/similarity_catalog.pt, models/compatibility_catalog.pt to
already exist (see scripts/train_*_final.py, scripts/build_*_catalog.py).

Run directly:
    python app/app.py
"""

from PIL import Image
import gradio as gr
import torch

from stylematch.catalog import load_catalog, search_by_compatibility, search_by_similarity
from stylematch.compatibility.dataset import CATEGORY_TO_LABEL, SEMANTIC_CATEGORIES
from stylematch.compatibility.model import CompatibilityModel
from stylematch.config import DEEPFASHION_DIR, MODELS_DIR, POLYVORE_DIR
from stylematch.similarity.model import SimilarityModel

TOP_K = 7
EMBEDDING_DIM = 256
FREEZE_UNTIL_BLOCK = 10

similarity_model = SimilarityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
similarity_model.load_state_dict(
    torch.load(MODELS_DIR / "similarity_final.pt", map_location=similarity_model.device)
)
similarity_catalog = load_catalog(MODELS_DIR / "similarity_catalog.pt")

compatibility_model = CompatibilityModel(embedding_dim=EMBEDDING_DIM, freeze_until_block=FREEZE_UNTIL_BLOCK)
compatibility_model.load_state_dict(
    torch.load(MODELS_DIR / "compatibility_final.pt", map_location=compatibility_model.device)
)
compatibility_catalog = load_catalog(MODELS_DIR / "compatibility_catalog.pt")


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


def find_compatible_labeled(image1, image2, image3, category_name: str) -> list:
    exclude = {CATEGORY_TO_LABEL[category_name]} if category_name else None
    return _find_compatible(_filter_none(image1, image2, image3), exclude_category_labels=exclude)


def _placeholder_results(image1, image2, image3) -> list:
    """Not wired to a model yet -- just echoes the upload(s) back so
    the gallery/layout can be checked end-to-end."""
    return _filter_none(image1, image2, image3)


def _placeholder_result_single(image) -> list:
    return _filter_none(image)


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
            f"Upload one, two, or three garments and tell us the NEW item's category, see the "
            f"top-{TOP_K} items to add, excluding items from that same category."
        )
        with gr.Row():
            compat_labeled_input1 = gr.Image(type="pil", label="Garment 1")
            compat_labeled_input2 = gr.Image(type="pil", label="Garment 2 (optional)")
            compat_labeled_input3 = gr.Image(type="pil", label="Garment 3 (optional)")
        compat_labeled_category = gr.Dropdown(choices=SEMANTIC_CATEGORIES, label="New item's category")
        compat_labeled_button = gr.Button("Find compatible items")
        compat_labeled_output = gr.Gallery(label="Top matches", columns=TOP_K, object_fit="contain", allow_preview=False)
        compat_labeled_button.click(
            fn=find_compatible_labeled,
            inputs=[compat_labeled_input1, compat_labeled_input2, compat_labeled_input3, compat_labeled_category],
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
            fn=_placeholder_results,
            inputs=[compat_infer_input1, compat_infer_input2, compat_infer_input3],
            outputs=compat_infer_output,
        )

    with gr.Tab("Stretch: African garments"):
        gr.Markdown(
            f"Upload an African garment, see its top-{TOP_K} similar AND top-{TOP_K} compatible "
            "items (category inferred the same way as the Infer tab). Stretch goal, time-permitting."
        )
        stretch_input = gr.Image(type="pil", label="Upload a garment")
        stretch_button = gr.Button("Find similar & compatible items")
        stretch_similar_output = gr.Gallery(label="Similar items", columns=TOP_K, object_fit="contain", allow_preview=False)
        stretch_compat_output = gr.Gallery(label="Compatible items", columns=TOP_K, object_fit="contain", allow_preview=False)
        stretch_button.click(
            fn=_placeholder_result_single, inputs=[stretch_input], outputs=stretch_similar_output
        )
        stretch_button.click(
            fn=_placeholder_result_single, inputs=[stretch_input], outputs=stretch_compat_output
        )


if __name__ == "__main__":
    demo.launch()
