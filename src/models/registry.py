"""Registry of supported generation, editing, and analysis models."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelCapability(Enum):
    GENERATE = "generate"
    EDIT = "edit"
    INPAINT = "inpaint"
    ANALYZE = "analyze"


@dataclass
class ModelConfig:
    name: str
    provider: str
    api_model_string: str
    capabilities: list[ModelCapability]
    max_resolution: tuple[int, int]
    supports_mask: bool = False
    cost_per_image: Optional[float] = None
    avg_latency_seconds: Optional[float] = None


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Generators
    "gemini": ModelConfig(
        name="Gemini 3 Pro Image Preview",
        provider="google",
        api_model_string="gemini-3-pro-image-preview",
        capabilities=[ModelCapability.GENERATE, ModelCapability.EDIT],
        max_resolution=(2048, 2048),
    ),
    "openai": ModelConfig(
        name="GPT Image 1",
        provider="openai",
        api_model_string="gpt-image-1",
        capabilities=[ModelCapability.GENERATE, ModelCapability.EDIT, ModelCapability.INPAINT],
        max_resolution=(4096, 4096),
        supports_mask=True,
    ),
    "flux": ModelConfig(
        name="FLUX.2 Pro",
        provider="fal",
        api_model_string="fal-ai/flux-2-pro",
        capabilities=[ModelCapability.GENERATE],
        max_resolution=(2048, 2048),
    ),
    "flux-edit": ModelConfig(
        name="FLUX.2 Pro Edit",
        provider="fal",
        api_model_string="fal-ai/flux-2-pro/edit",
        capabilities=[ModelCapability.EDIT],
        max_resolution=(2048, 2048),
    ),
    "recraft": ModelConfig(
        name="Recraft V3",
        provider="fal",
        api_model_string="fal-ai/recraft/v3/text-to-image",
        capabilities=[ModelCapability.GENERATE],
        max_resolution=(2048, 2048),
    ),
    "recraft-edit": ModelConfig(
        name="Recraft V3 Edit",
        provider="fal",
        api_model_string="fal-ai/recraft/v3/image-to-image",
        capabilities=[ModelCapability.EDIT],
        max_resolution=(2048, 2048),
    ),

    # Analyzers (VLMs)
    "claude": ModelConfig(
        name="Claude 4.5 Opus",
        provider="anthropic",
        api_model_string="claude-opus-4-5",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(8192, 8192),
    ),
    "gpt5": ModelConfig(
        name="GPT-5.1",
        provider="openai",
        api_model_string="gpt-5.1",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
    "gemini-vlm": ModelConfig(
        name="Gemini 3 Pro Preview",
        provider="google",
        api_model_string="gemini-3-pro-preview",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
    "qwen": ModelConfig(
        name="Qwen3 VL 235B",
        provider="openrouter",
        api_model_string="qwen/qwen3-vl-235b-a22b-instruct",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
}

__all__ = [
    "ModelCapability",
    "ModelConfig",
    "MODEL_REGISTRY",
]
