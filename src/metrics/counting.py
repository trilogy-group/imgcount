"""Metrics for evaluating object counting accuracy."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CountingResult:
    """Single counting evaluation result."""

    prompt: str
    target_count: int
    predicted_count: int
    object_type: str
    iteration: int
    generator: str
    analyzer: str
    image_path: str
    vlm_confidence: Optional[float] = None
    vlm_reasoning: Optional[str] = None


@dataclass
class CountingMetrics:
    """Aggregated counting metrics."""

    counting_accuracy_rate: float
    mean_absolute_error: float
    mean_signed_error: float
    within_one_accuracy: float
    undercount_rate: float
    overcount_rate: float
    n_samples: int


def compute_counting_metrics(results: List[CountingResult]) -> CountingMetrics:
    """Compute all counting metrics from a list of results."""
    if not results:
        raise ValueError("Empty results list")

    targets = np.array([r.target_count for r in results])
    predictions = np.array([r.predicted_count for r in results])
    errors = predictions - targets

    return CountingMetrics(
        counting_accuracy_rate=float(np.mean(predictions == targets)),
        mean_absolute_error=float(np.mean(np.abs(errors))),
        mean_signed_error=float(np.mean(errors)),
        within_one_accuracy=float(np.mean(np.abs(errors) <= 1)),
        undercount_rate=float(np.mean(errors < 0)),
        overcount_rate=float(np.mean(errors > 0)),
        n_samples=len(results),
    )


def compute_metrics_by_group(
    results: List[CountingResult],
    group_by: str,
) -> dict[str, CountingMetrics]:
    """Compute metrics grouped by a specified attribute."""
    from collections import defaultdict

    if not results:
        raise ValueError("Empty results list")

    groups = defaultdict(list)
    for r in results:
        key = getattr(r, group_by)
        groups[key].append(r)

    return {k: compute_counting_metrics(v) for k, v in groups.items()}


def confusion_matrix(results: List[CountingResult], max_count: int = 10) -> np.ndarray:
    """Generate confusion matrix for target vs predicted counts."""
    matrix = np.zeros((max_count + 1, max_count + 1), dtype=int)
    for r in results:
        if r.target_count <= max_count and r.predicted_count <= max_count:
            matrix[r.target_count, r.predicted_count] += 1
    return matrix


__all__ = [
    "CountingResult",
    "CountingMetrics",
    "compute_counting_metrics",
    "compute_metrics_by_group",
    "confusion_matrix",
]
