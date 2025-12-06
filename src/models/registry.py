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
        name="Gemini 3 Pro Image",
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
        max_resolution=(1792, 1792),
        supports_mask=True,
    ),
    "recraft": ModelConfig(
        name="Recraft V3.5",
        provider="fal",
        api_model_string="recraft-v3.5",
        capabilities=[ModelCapability.GENERATE, ModelCapability.EDIT],
        max_resolution=(2048, 2048),
    ),
    "flux": ModelConfig(
        name="FLUX.2 Pro",
        provider="fal",
        api_model_string="fal-ai/flux-2-pro",
        capabilities=[ModelCapability.GENERATE],
        max_resolution=(2048, 2048),
    ),
    "flux-pro-edit": ModelConfig(
        name="FLUX.2 Pro Edit",
        provider="fal",
        api_model_string="fal-ai/flux-2-pro/edit",
        capabilities=[ModelCapability.EDIT],
        max_resolution=(2048, 2048),
        supports_mask=True,
    ),
    "flux-fill": ModelConfig(
        name="FLUX.2 Fill",
        provider="fal",
        api_model_string="fal-ai/flux-2-fill",
        capabilities=[ModelCapability.INPAINT],
        max_resolution=(2048, 2048),
        supports_mask=True,
    ),
    "ideogram": ModelConfig(
        name="Ideogram V3",
        provider="ideogram",
        api_model_string="ideogram-v3",
        capabilities=[ModelCapability.GENERATE],
        max_resolution=(1536, 1536),
    ),
    "sd35": ModelConfig(
        name="Stable Diffusion 3.5 Large",
        provider="stability",
        api_model_string="sd-3.5-large",
        capabilities=[ModelCapability.GENERATE],
        max_resolution=(2048, 2048),
    ),

    # Analyzers
    "qwen": ModelConfig(
        name="Qwen3 VL 235B",
        provider="openrouter",
        api_model_string="qwen/qwen3-vl-235b",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
    "gemini-analyzer": ModelConfig(
        name="Gemini 3 Pro",
        provider="google",
        api_model_string="gemini-3-pro",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
    "gpt5": ModelConfig(
        name="GPT-5.1",
        provider="openai",
        api_model_string="gpt-5.1",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
    "claude": ModelConfig(
        name="Claude 4.5 Opus",
        provider="anthropic",
        api_model_string="claude-opus-4-5-20251101",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(8192, 8192),
    ),
    "llama4": ModelConfig(
        name="Llama 4 Maverick",
        provider="openrouter",
        api_model_string="meta-llama/llama-4-maverick",
        capabilities=[ModelCapability.ANALYZE],
        max_resolution=(4096, 4096),
    ),
}

__all__ = [
    "ModelCapability",
    "ModelConfig",
    "MODEL_REGISTRY",
]
