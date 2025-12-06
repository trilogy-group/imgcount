"""Aggregation utilities for metrics results."""
from dataclasses import asdict
from typing import Dict, List, Optional

from .counting import (
    CountingMetrics,
    CountingResult,
    compute_counting_metrics,
    compute_metrics_by_group,
    confusion_matrix,
)


def summarize_counting_results(
    results: List[CountingResult],
    max_count: int = 10,
    group_by: Optional[str] = None,
) -> Dict[str, object]:
    """Produce a summary dictionary for counting results."""
    summary: Dict[str, object] = {
        "overall": asdict(compute_counting_metrics(results)),
        "confusion_matrix": confusion_matrix(results, max_count=max_count).tolist(),
    }

    if group_by:
        grouped = compute_metrics_by_group(results, group_by=group_by)
        summary["grouped"] = {key: asdict(value) for key, value in grouped.items()}

    return summary


__all__ = ["summarize_counting_results"]
