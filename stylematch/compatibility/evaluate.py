"""Evaluation for the compatibility engine: FITB accuracy and
compatibility AUC, the same two metrics Vasileva et al. report (see
references/polyvore_outfits_schema.md and
reports/figures/05_compatibility_engine_metrics_mockup.png's
illustrative placeholders), so real numbers from here stay directly
comparable to that citation.
"""

import torch
from sklearn.metrics import roc_auc_score

from stylematch.compatibility.dataset import CompatibilityDataset, FITBDataset
from stylematch.compatibility.model import CompatibilityModel


@torch.no_grad()
def fitb_accuracy(model: CompatibilityModel, dataset: FITBDataset) -> float:
    """For each FITB question, scores all 4 candidates (context +
    each candidate, via score_outfit) and checks whether the model's
    highest-scoring candidate is the correct one (always index 0, see
    dataset.py). Context is embedded once per question, not once per
    candidate -- the 4 candidates share the same context, so
    re-running the backbone on it 4 times would be pure waste.
    """
    model.eval()
    correct = 0
    for idx in range(len(dataset)):
        context_images, candidate_images, correct_index = dataset[idx]
        context_embeddings = model(torch.stack(context_images))
        candidate_embeddings = model(torch.stack(candidate_images))

        scores = [
            model.score_outfit(torch.cat([context_embeddings, candidate_embeddings[c : c + 1]])).item()
            for c in range(len(candidate_images))
        ]
        predicted_index = max(range(len(scores)), key=lambda i: scores[i])
        if predicted_index == correct_index:
            correct += 1
    return correct / len(dataset)


@torch.no_grad()
def compatibility_auc(model: CompatibilityModel, dataset: CompatibilityDataset) -> float:
    """Scores every candidate outfit in `dataset` (real and generated
    negatives) via score_outfit(), then computes ROC-AUC between those
    scores and the real/fake labels -- how well the model's raw score
    separates compatible from incompatible outfits, across every
    possible threshold at once, not just a single pass/fail cutoff.
    """
    model.eval()
    scores = []
    labels = []
    for idx in range(len(dataset)):
        outfit_images, label = dataset[idx]
        embeddings = model(torch.stack(outfit_images))
        scores.append(model.score_outfit(embeddings).item())
        labels.append(label)
    return roc_auc_score(labels, scores)


def evaluate_compatibility(
    model: CompatibilityModel,
    split: str = "valid",
    max_fitb_questions: int | None = None,
    max_compat_outfits: int | None = None,
) -> dict[str, float]:
    """Computes both real evaluation metrics against `split`: FITB
    accuracy and compatibility AUC. Returns a dict directly usable to
    replace Figure 5's illustrative placeholder numbers with real ones.

    Args:
        max_fitb_questions: caps how many FITB questions to score, in
            dataset order. None (default) evaluates every question in
            the split -- the full valid split is 5,000 questions, so
            capping this is useful for a quick local check before a
            full-scale run.
        max_compat_outfits: same capping, for the compatibility (AUC)
            task's candidate outfits. Balanced across both labels, not
            a plain first-N slice: the real file is grouped by label
            (all label=1 rows first, then label=0, confirmed via the
            real data, see test_compatibility_dataset.py), so a plain
            slice could silently grab only one label and break AUC
            (which needs both classes present).
    """
    fitb_dataset = FITBDataset(split=split, transform=model.preprocess_val)
    compat_dataset = CompatibilityDataset(split=split, transform=model.preprocess_val)
    if max_fitb_questions is not None:
        fitb_dataset.questions = fitb_dataset.questions[:max_fitb_questions]
    if max_compat_outfits is not None:
        per_label_cap = max_compat_outfits // 2
        positives = [q for q in compat_dataset.questions if q[0] == 1][:per_label_cap]
        negatives = [q for q in compat_dataset.questions if q[0] == 0][:per_label_cap]
        compat_dataset.questions = positives + negatives

    return {
        "fitb_accuracy": fitb_accuracy(model, fitb_dataset),
        "fitb_num_questions": len(fitb_dataset),
        "compatibility_auc": compatibility_auc(model, compat_dataset),
        "compatibility_num_outfits": len(compat_dataset),
    }
