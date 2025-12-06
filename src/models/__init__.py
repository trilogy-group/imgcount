from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ImageResult:
    image_path: str
    metadata: dict[str, Any]


class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> ImageResult:
        """Generates an image based on the prompt."""
        raise NotImplementedError


class ImageEditor(ABC):
    @abstractmethod
    def edit(self, image_path: str, prompt: str) -> ImageResult:
        """Edits an existing image based on the prompt."""
        raise NotImplementedError


class ImageAnalyzer(ABC):
    @abstractmethod
    def analyze(self, image_path: str, prompt: str) -> int:
        """Analyzes an image and returns the count of objects."""
        raise NotImplementedError


__all__ = [
    "ImageResult",
    "ImageGenerator",
    "ImageEditor",
    "ImageAnalyzer",
]
