"""Image quality assessment metrics."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class QualityScore:
    """Quality scores for a single image."""

    image_path: str
    clip_iqa: Optional[float] = None
    topiq: Optional[float] = None
    aesthetic: Optional[float] = None
    artifact_score: Optional[float] = None


@dataclass
class QualityDegradationResult:
    """Track quality across correction iterations."""

    prompt: str
    iteration_scores: List[QualityScore]
    quality_degradation_index: float


class QualityAssessor:
    """Wrapper for image quality assessment models."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_iqa = None
        self._topiq = None

    def _load_clip_iqa(self):
        """Lazy load CLIP-IQA model."""
        if self._clip_iqa is None:
            import pyiqa

            self._clip_iqa = pyiqa.create_metric("clipiqa", device=self.device)
        return self._clip_iqa

    def _load_topiq(self):
        """Lazy load TOPIQ model."""
        if self._topiq is None:
            import pyiqa

            self._topiq = pyiqa.create_metric("topiq_nr", device=self.device)
        return self._topiq

    def assess(self, image_path: str) -> QualityScore:
        """Compute all quality metrics for an image."""
        from PIL import Image

        Image.open(image_path).convert("RGB")

        clip_score = float(self._load_clip_iqa()(image_path))
        topiq_score = float(self._load_topiq()(image_path))

        return QualityScore(
            image_path=image_path,
            clip_iqa=clip_score,
            topiq=topiq_score,
        )

    def compute_degradation_index(self, image_sequence: List[str]) -> float:
        """
        Compute quality degradation across iteration sequence.
        Returns: Negative value indicates degradation.
        """
        if len(image_sequence) < 2:
            return 0.0

        scores = [self.assess(path).clip_iqa for path in image_sequence]
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        return float(slope)


__all__ = [
    "QualityScore",
    "QualityDegradationResult",
    "QualityAssessor",
]
