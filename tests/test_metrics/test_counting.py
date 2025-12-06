import pytest

from src.metrics.counting import (
    CountingResult,
    compute_counting_metrics,
    confusion_matrix,
)


@pytest.fixture
def sample_results():
    return [
        CountingResult(
            prompt="3 apples",
            target_count=3,
            predicted_count=3,
            object_type="apples",
            iteration=0,
            generator="gen-a",
            analyzer="vlm-a",
            image_path="/tmp/a.png",
        ),
        CountingResult(
            prompt="5 oranges",
            target_count=5,
            predicted_count=4,
            object_type="oranges",
            iteration=0,
            generator="gen-a",
            analyzer="vlm-a",
            image_path="/tmp/b.png",
        ),
        CountingResult(
            prompt="2 bananas",
            target_count=2,
            predicted_count=2,
            object_type="bananas",
            iteration=1,
            generator="gen-b",
            analyzer="vlm-b",
            image_path="/tmp/c.png",
        ),
        CountingResult(
            prompt="7 lemons",
            target_count=7,
            predicted_count=9,
            object_type="lemons",
            iteration=1,
            generator="gen-b",
            analyzer="vlm-b",
            image_path="/tmp/d.png",
        ),
    ]


def test_counting_accuracy_rate(sample_results):
    metrics = compute_counting_metrics(sample_results)
    assert metrics.counting_accuracy_rate == 0.5


def test_mean_absolute_error(sample_results):
    metrics = compute_counting_metrics(sample_results)
    assert metrics.mean_absolute_error == 0.75


def test_confusion_matrix_shape(sample_results):
    matrix = confusion_matrix(sample_results, max_count=10)
    assert matrix.shape == (11, 11)
