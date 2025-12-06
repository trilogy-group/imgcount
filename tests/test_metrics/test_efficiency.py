from pytest import approx

from src.metrics.efficiency import (
    LoopResult,
    compute_efficiency_metrics,
    correction_efficiency_score,
)


def test_efficiency_metrics_basic():
    results = [
        LoopResult(
            prompt="3 apples",
            target_count=3,
            initial_count=2,
            final_count=3,
            iterations_used=2,
            max_iterations=3,
            success=True,
            total_time_seconds=5.0,
            cost_usd=0.1,
        ),
        LoopResult(
            prompt="5 oranges",
            target_count=5,
            initial_count=5,
            final_count=5,
            iterations_used=1,
            max_iterations=3,
            success=True,
            total_time_seconds=3.0,
            cost_usd=0.05,
        ),
        LoopResult(
            prompt="2 bananas",
            target_count=2,
            initial_count=4,
            final_count=3,
            iterations_used=3,
            max_iterations=3,
            success=False,
            total_time_seconds=6.0,
            cost_usd=None,
        ),
    ]

    metrics = compute_efficiency_metrics(results)

    assert metrics.success_rate == approx(2 / 3)
    assert metrics.first_pass_success_rate == approx(1 / 3)
    assert metrics.mean_iterations_all == approx((2 + 1 + 3) / 3)
    assert metrics.mean_cost_usd == approx(0.075)

    score = correction_efficiency_score(results)
    assert 0 <= score <= 1
