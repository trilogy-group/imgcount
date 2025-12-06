from src.metrics.quality import QualityAssessor, QualityScore


def test_compute_degradation_index(monkeypatch):
    assessor = QualityAssessor(device="cpu")
    scores = iter([0.9, 0.8, 0.7])

    def fake_assess(path: str) -> QualityScore:
        return QualityScore(image_path=path, clip_iqa=next(scores))

    monkeypatch.setattr(assessor, "assess", fake_assess)

    idx = assessor.compute_degradation_index(["a.png", "b.png", "c.png"])
    assert idx < 0
