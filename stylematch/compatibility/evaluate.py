"""Evaluation for the compatibility engine: FITB accuracy and
compatibility AUC, the same two metrics Vasileva et al. report.
"""

import torch
from sklearn.metrics import roc_auc_score

from stylematch.compatibility.dataset import CompatibilityDataset, FITBDataset
from stylematch.compatibility.model import CompatibilityModel


@torch.no_grad()
def fitb_accuracy(model: CompatibilityModel, dataset: FITBDataset) -> float:
    """For each FITB question, scores all 4 candidates (context +
    candidate, via score_outfit) and checks the top-scoring one is
    correct (always index 0). Context embedded once per question, not
    once per candidate, since all 4 share it."""
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
    """Scores every candidate outfit (real + generated negatives) via
    score_outfit(), then computes ROC-AUC against the real/fake
    labels -- how well the raw score separates them across thresholds."""
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
    """Computes both evaluation metrics against `split`: FITB accuracy
    and compatibility AUC.

    Args:
        max_fitb_questions: caps questions scored, in dataset order.
            None evaluates the full split (5,000 questions).
        max_compat_outfits: same capping, but balanced across labels
            (not a plain first-N slice) -- the real file is grouped by
            label, so a plain slice could grab only one and break AUC.
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
