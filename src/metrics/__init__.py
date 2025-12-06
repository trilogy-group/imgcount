"""Metrics package for imgcount."""
from .counting import (
    CountingMetrics,
    CountingResult,
    compute_counting_metrics,
    compute_metrics_by_group,
    confusion_matrix,
)
from .quality import QualityAssessor, QualityDegradationResult, QualityScore
from .efficiency import (
    EfficiencyMetrics,
    LoopResult,
    compute_efficiency_metrics,
    correction_efficiency_score,
)
from .agreement import (
    AgreementResult,
    HumanAnnotation,
    VLMPrediction,
    compute_vlm_agreement,
    inter_vlm_agreement,
)
from .aggregation import summarize_counting_results

__all__ = [
    "CountingMetrics",
    "CountingResult",
    "compute_counting_metrics",
    "compute_metrics_by_group",
    "confusion_matrix",
    "QualityAssessor",
    "QualityDegradationResult",
    "QualityScore",
    "EfficiencyMetrics",
    "LoopResult",
    "compute_efficiency_metrics",
    "correction_efficiency_score",
    "AgreementResult",
    "HumanAnnotation",
    "VLMPrediction",
    "compute_vlm_agreement",
    "inter_vlm_agreement",
    "summarize_counting_results",
]
