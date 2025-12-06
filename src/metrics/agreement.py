"""Inter-rater agreement metrics for VLM evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class AgreementResult:
    """Agreement between VLM predictions and ground truth."""

    vlm_name: str
    accuracy: float
    pearson_correlation: float
    spearmans_rho: float
    cohens_kappa: float
    mean_absolute_error: float


@dataclass
class VLMPrediction:
    """Single VLM prediction for an image."""

    image_id: str
    vlm_name: str
    predicted_count: int
    confidence: Optional[float] = None


@dataclass
class HumanAnnotation:
    """Human ground truth for an image."""

    image_id: str
    true_count: int
    annotator_counts: List[int]


def compute_vlm_agreement(
    predictions: List[VLMPrediction],
    ground_truth: Dict[str, HumanAnnotation],
) -> Dict[str, AgreementResult]:
    """Compute agreement metrics for each VLM against human ground truth."""
    from collections import defaultdict
    from sklearn.metrics import cohen_kappa_score

    vlm_preds = defaultdict(list)
    vlm_truths = defaultdict(list)

    for pred in predictions:
        if pred.image_id in ground_truth:
            vlm_preds[pred.vlm_name].append(pred.predicted_count)
            vlm_truths[pred.vlm_name].append(ground_truth[pred.image_id].true_count)

    results: Dict[str, AgreementResult] = {}
    for vlm_name in vlm_preds:
        preds = np.array(vlm_preds[vlm_name])
        truths = np.array(vlm_truths[vlm_name])

        preds_clipped = np.clip(preds, 0, 10)
        truths_clipped = np.clip(truths, 0, 10)

        results[vlm_name] = AgreementResult(
            vlm_name=vlm_name,
            accuracy=float(np.mean(preds == truths)) if truths.size else 0.0,
            pearson_correlation=float(stats.pearsonr(preds, truths)[0]) if len(truths) > 1 else 0.0,
            spearmans_rho=float(stats.spearmanr(preds, truths)[0]) if len(truths) > 1 else 0.0,
            cohens_kappa=float(cohen_kappa_score(truths_clipped, preds_clipped)) if len(truths_clipped) > 0 else 0.0,
            mean_absolute_error=float(np.mean(np.abs(preds - truths))) if truths.size else 0.0,
        )

    return results


def inter_vlm_agreement(predictions: List[VLMPrediction]) -> float:
    """Compute Fleiss' kappa for agreement among multiple VLMs."""
    from collections import defaultdict
    from statsmodels.stats.inter_rater import fleiss_kappa

    if not predictions:
        return 0.0

    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred.image_id].append(pred.predicted_count)

    max_count = 10
    n_images = len(by_image)
    rating_matrix = np.zeros((n_images, max_count + 1))

    for i, counts in enumerate(by_image.values()):
        for count in counts:
            if 0 <= count <= max_count:
                rating_matrix[i, count] += 1

    return float(fleiss_kappa(rating_matrix)) if n_images else 0.0


__all__ = [
    "AgreementResult",
    "VLMPrediction",
    "HumanAnnotation",
    "compute_vlm_agreement",
    "inter_vlm_agreement",
]
