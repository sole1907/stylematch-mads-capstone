"""StyleMatch demo app, UI skeleton only -- no model inference wired
yet (that's the next pass). Each tab's button currently just echoes
the upload(s) back into the results gallery, to confirm the layout
and data flow work before any real model call exists.

5 tabs, matching the planned build order:
1. Similarity: upload 1 image -> top-7 similar items.
2. Compatibility (naive): upload 1 garment -> top-7 compatible items,
   no category filtering.
3. Compatibility (labeled): same, plus a category dropdown (the
   user states the garment's category) to filter same-category
   candidates out before ranking.
4. Compatibility (infer): same, but the category is inferred from
   the similarity engine's nearest neighbors instead of asked for.
5. Stretch goal: African garments, both engines, category inferred
   the same way as tab 4.

Run directly:
    python app/app.py
"""

import gradio as gr

from stylematch.compatibility.dataset import SEMANTIC_CATEGORIES

TOP_K = 7


def _placeholder_results(*images) -> list:
    """Not wired to a model yet -- just echoes the upload(s) back so
    the gallery/layout can be checked end-to-end."""
    return [img for img in images if img is not None]


with gr.Blocks(title="StyleMatch") as demo:
    gr.Markdown("# StyleMatch — fashion similarity & compatibility demo")

    with gr.Tab("Similarity"):
        gr.Markdown(f"Upload a garment, see its top-{TOP_K} most similar items.")
        similarity_input = gr.Image(type="pil", label="Upload a garment")
        similarity_button = gr.Button("Find similar items")
        similarity_output = gr.Gallery(label="Top matches", columns=TOP_K)
        similarity_button.click(
            fn=_placeholder_results, inputs=[similarity_input], outputs=similarity_output
        )

    with gr.Tab("Compatibility (naive)"):
        gr.Markdown(f"Upload a garment, see its top-{TOP_K} compatible items. No category filtering.")
        compat_naive_input = gr.Image(type="pil", label="Upload a garment")
        compat_naive_button = gr.Button("Find compatible items")
        compat_naive_output = gr.Gallery(label="Top matches", columns=TOP_K)
        compat_naive_button.click(
            fn=_placeholder_results, inputs=[compat_naive_input], outputs=compat_naive_output
        )

    with gr.Tab("Compatibility (labeled)"):
        gr.Markdown(
            f"Upload a garment and tell us its category, see its top-{TOP_K} compatible items, "
            "excluding items from the same category."
        )
        compat_labeled_input = gr.Image(type="pil", label="Upload a garment")
        compat_labeled_category = gr.Dropdown(choices=SEMANTIC_CATEGORIES, label="Garment category")
        compat_labeled_button = gr.Button("Find compatible items")
        compat_labeled_output = gr.Gallery(label="Top matches", columns=TOP_K)
        compat_labeled_button.click(
            fn=_placeholder_results, inputs=[compat_labeled_input], outputs=compat_labeled_output
        )

    with gr.Tab("Compatibility (infer)"):
        gr.Markdown(
            f"Upload a garment, category inferred automatically (via the similarity engine's "
            f"nearest neighbors), see its top-{TOP_K} compatible items. Exploratory -- category "
            "inference isn't separately validated."
        )
        compat_infer_input = gr.Image(type="pil", label="Upload a garment")
        compat_infer_button = gr.Button("Find compatible items")
        compat_infer_category = gr.Textbox(label="Inferred category", interactive=False)
        compat_infer_output = gr.Gallery(label="Top matches", columns=TOP_K)
        compat_infer_button.click(
            fn=_placeholder_results, inputs=[compat_infer_input], outputs=compat_infer_output
        )

    with gr.Tab("Stretch: African garments"):
        gr.Markdown(
            f"Upload an African garment, see its top-{TOP_K} similar AND top-{TOP_K} compatible "
            "items (category inferred the same way as the Infer tab). Stretch goal, time-permitting."
        )
        stretch_input = gr.Image(type="pil", label="Upload a garment")
        stretch_button = gr.Button("Find similar & compatible items")
        stretch_similar_output = gr.Gallery(label="Similar items", columns=TOP_K)
        stretch_compat_output = gr.Gallery(label="Compatible items", columns=TOP_K)
        stretch_button.click(
            fn=_placeholder_results, inputs=[stretch_input], outputs=stretch_similar_output
        )
        stretch_button.click(
            fn=_placeholder_results, inputs=[stretch_input], outputs=stretch_compat_output
        )


if __name__ == "__main__":
    demo.launch()
