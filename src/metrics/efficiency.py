"""Metrics for correction loop efficiency."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LoopResult:
    """Result of a single correction loop run."""

    prompt: str
    target_count: int
    initial_count: int
    final_count: int
    iterations_used: int
    max_iterations: int
    success: bool
    total_time_seconds: float
    cost_usd: Optional[float] = None


@dataclass
class EfficiencyMetrics:
    """Aggregated efficiency metrics."""

    success_rate: float
    mean_iterations_to_success: float
    mean_iterations_all: float
    first_pass_success_rate: float
    correction_improvement_rate: float
    mean_time_seconds: float
    mean_cost_usd: Optional[float]


def compute_efficiency_metrics(results: List[LoopResult]) -> EfficiencyMetrics:
    """Compute efficiency metrics from loop results."""
    successes = [r for r in results if r.success]
    initial_correct = [r for r in results if r.initial_count == r.target_count]
    initial_wrong = [r for r in results if r.initial_count != r.target_count]
    corrected = [r for r in initial_wrong if r.success]

    return EfficiencyMetrics(
        success_rate=len(successes) / len(results) if results else 0,
        mean_iterations_to_success=float(np.mean([r.iterations_used for r in successes])) if successes else 0,
        mean_iterations_all=float(np.mean([r.iterations_used for r in results])) if results else 0,
        first_pass_success_rate=len(initial_correct) / len(results) if results else 0,
        correction_improvement_rate=len(corrected) / len(initial_wrong) if initial_wrong else 0,
        mean_time_seconds=float(np.mean([r.total_time_seconds for r in results])) if results else 0,
        mean_cost_usd=float(np.mean([r.cost_usd for r in results if r.cost_usd])) if any(r.cost_usd for r in results) else None,
    )


def correction_efficiency_score(results: List[LoopResult]) -> float:
    """
    Compute CES: Success weighted by iteration efficiency.
    CES = mean(success * (max_iter - iter_used + 1) / max_iter)
    Higher is better. Max = 1.0 (all succeed on first try)
    """
    scores: List[float] = []
    for r in results:
        if r.success:
            efficiency = (r.max_iterations - r.iterations_used + 1) / r.max_iterations
        else:
            efficiency = 0
        scores.append(efficiency)
    return float(np.mean(scores)) if scores else 0.0


__all__ = [
    "LoopResult",
    "EfficiencyMetrics",
    "compute_efficiency_metrics",
    "correction_efficiency_score",
]
