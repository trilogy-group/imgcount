from src.metrics.agreement import (
    AgreementResult,
    HumanAnnotation,
    VLMPrediction,
    compute_vlm_agreement,
    inter_vlm_agreement,
)


def test_compute_vlm_agreement():
    ground_truth = {
        "img1": HumanAnnotation(image_id="img1", true_count=3, annotator_counts=[3, 3, 3]),
        "img2": HumanAnnotation(image_id="img2", true_count=5, annotator_counts=[5, 4, 5]),
    }
    predictions = [
        VLMPrediction(image_id="img1", vlm_name="vlm-a", predicted_count=3),
        VLMPrediction(image_id="img2", vlm_name="vlm-a", predicted_count=4),
        VLMPrediction(image_id="img1", vlm_name="vlm-b", predicted_count=2),
        VLMPrediction(image_id="img2", vlm_name="vlm-b", predicted_count=5),
    ]

    results = compute_vlm_agreement(predictions, ground_truth)

    assert set(results.keys()) == {"vlm-a", "vlm-b"}
    assert all(isinstance(v, AgreementResult) for v in results.values())


def test_inter_vlm_agreement():
    predictions = [
        VLMPrediction(image_id="img1", vlm_name="vlm-a", predicted_count=1),
        VLMPrediction(image_id="img1", vlm_name="vlm-b", predicted_count=1),
        VLMPrediction(image_id="img2", vlm_name="vlm-a", predicted_count=2),
        VLMPrediction(image_id="img2", vlm_name="vlm-b", predicted_count=3),
    ]

    kappa = inter_vlm_agreement(predictions)
    assert isinstance(kappa, float)
