# imgcount Enhancement Specification

## Project Overview

**Repository:** https://github.com/trilogy-group/imgcount  
**Purpose:** Enhance the image generation and analysis platform to support comprehensive evaluation of generative numeracy (object counting accuracy) with VLM-in-the-loop quality assessment.

**Target Paper:** 5th Workshop on Image/Video/Audio Quality Assessment in Computer Vision, VLM and Diffusion Model (WACV 2026)  
**Submission Deadline:** December 15, 2025

---

## Current Architecture Summary

The existing codebase implements:
- Direct image generation via multiple T2I models
- VLM-based object counting analysis
- Auto-correction loop (generate → analyze → edit if wrong)
- CLI interface via `main.py`

**Current Directory Structure:**
```
imgcount/
├── src/
├── tests/
├── docs/
├── main.py
├── pyproject.toml
└── uv.lock
```

---

## Enhancement Requirements

### 1. Model Updates

Update all model integrations to use current SOTA versions.

#### 1.1 Generator Models

| Current | Updated | Provider | Notes |
|---------|---------|----------|-------|
| Gemini 2.5 Flash Image | `gemini-3-pro-image-preview` | Google AI | Native image generation |
| GPT Image 1 | `gpt-image-1` | OpenAI | Latest DALL-E backend |
| Recraft V3 | `fal-ai/recraft/v3/text-to-image` | Fal.ai | Text-to-image generation |
| - | `fal-ai/flux-2-pro` | Fal.ai / Replicate | Add FLUX.2 support |
| - | `ideogram-v3` | Ideogram API | Add Ideogram support |
| - | `sd-3.5-large` | Stability AI | Add SD 3.5 support |

#### 1.2 Editor Models

| Current | Updated | Provider | Notes |
|---------|---------|----------|-------|
| GPT Image 1 (edit) | `gpt-image-1` | OpenAI | Image editing endpoint |
| Recraft V3 (edit) | `fal-ai/recraft/v3/image-to-image` | Fal.ai | Image-to-image editing |
| - | `fal-ai/flux-2-pro/edit` | Fal.ai / Replicate | Add FLUX.2 edit support |

#### 1.3 Analyzer Models (VLMs)

| Current | Updated | Provider | Notes |
|---------|---------|----------|-------|
| Qwen3 VL 235B | `qwen/qwen3-vl-235b-a22b-instruct` | OpenRouter | Keep current |
| Gemini 3 Pro | `gemini-3-pro-preview` | Google AI | Vision capabilities |
| - | `gpt-5.1` | OpenAI | Add GPT-5.1 vision |
| - | `claude-opus-4-5` | Anthropic | Add Claude 4.5 Opus |

#### 1.4 Implementation Requirements

Create a model registry system:

```python
# src/models/registry.py

from enum import Enum
from dataclasses import dataclass
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
    cost_per_image: Optional[float] = None  # USD estimate
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
```

---

### 2. New Metrics Module

Create `src/metrics/` directory with comprehensive evaluation metrics.

#### 2.1 File Structure

```
src/metrics/
├── __init__.py
├── counting.py          # Counting-specific metrics
├── quality.py           # Image quality metrics
├── efficiency.py        # Loop efficiency metrics
├── agreement.py         # Inter-rater agreement
└── aggregation.py       # Result aggregation utilities
```

#### 2.2 Counting Metrics (`src/metrics/counting.py`)

```python
"""Metrics for evaluating object counting accuracy."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class CountingResult:
    """Single counting evaluation result."""
    prompt: str
    target_count: int
    predicted_count: int
    object_type: str
    iteration: int
    generator: str
    analyzer: str
    image_path: str
    vlm_confidence: Optional[float] = None
    vlm_reasoning: Optional[str] = None

@dataclass
class CountingMetrics:
    """Aggregated counting metrics."""
    counting_accuracy_rate: float  # Fraction exactly correct
    mean_absolute_error: float     # Average |pred - target|
    mean_signed_error: float       # Average (pred - target), shows bias
    within_one_accuracy: float     # Fraction within ±1
    undercount_rate: float         # Fraction pred < target
    overcount_rate: float          # Fraction pred > target
    n_samples: int

def compute_counting_metrics(results: List[CountingResult]) -> CountingMetrics:
    """Compute all counting metrics from a list of results."""
    if not results:
        raise ValueError("Empty results list")
    
    targets = np.array([r.target_count for r in results])
    predictions = np.array([r.predicted_count for r in results])
    errors = predictions - targets
    
    return CountingMetrics(
        counting_accuracy_rate=np.mean(predictions == targets),
        mean_absolute_error=np.mean(np.abs(errors)),
        mean_signed_error=np.mean(errors),
        within_one_accuracy=np.mean(np.abs(errors) <= 1),
        undercount_rate=np.mean(errors < 0),
        overcount_rate=np.mean(errors > 0),
        n_samples=len(results),
    )

def compute_metrics_by_group(
    results: List[CountingResult],
    group_by: str  # "target_count", "object_type", "generator", "analyzer"
) -> dict[str, CountingMetrics]:
    """Compute metrics grouped by a specified attribute."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for r in results:
        key = getattr(r, group_by)
        groups[key].append(r)
    
    return {k: compute_counting_metrics(v) for k, v in groups.items()}

def confusion_matrix(
    results: List[CountingResult],
    max_count: int = 10
) -> np.ndarray:
    """Generate confusion matrix for target vs predicted counts."""
    matrix = np.zeros((max_count + 1, max_count + 1), dtype=int)
    for r in results:
        if r.target_count <= max_count and r.predicted_count <= max_count:
            matrix[r.target_count, r.predicted_count] += 1
    return matrix
```

#### 2.3 Quality Metrics (`src/metrics/quality.py`)

```python
"""Image quality assessment metrics."""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import numpy as np

@dataclass 
class QualityScore:
    """Quality scores for a single image."""
    image_path: str
    clip_iqa: Optional[float] = None      # CLIP-IQA score
    topiq: Optional[float] = None         # TOPIQ score
    aesthetic: Optional[float] = None     # Aesthetic predictor
    artifact_score: Optional[float] = None # Artifact detection

@dataclass
class QualityDegradationResult:
    """Track quality across correction iterations."""
    prompt: str
    iteration_scores: List[QualityScore]
    quality_degradation_index: float  # Negative = degradation

class QualityAssessor:
    """Wrapper for image quality assessment models."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_iqa = None
        self._topiq = None
    
    def _load_clip_iqa(self):
        """Lazy load CLIP-IQA model."""
        if self._clip_iqa is None:
            # Use pyiqa library
            import pyiqa
            self._clip_iqa = pyiqa.create_metric('clipiqa', device=self.device)
        return self._clip_iqa
    
    def _load_topiq(self):
        """Lazy load TOPIQ model."""
        if self._topiq is None:
            import pyiqa
            self._topiq = pyiqa.create_metric('topiq_nr', device=self.device)
        return self._topiq
    
    def assess(self, image_path: str) -> QualityScore:
        """Compute all quality metrics for an image."""
        from PIL import Image
        import torch
        
        img = Image.open(image_path).convert('RGB')
        
        clip_score = float(self._load_clip_iqa()(image_path))
        topiq_score = float(self._load_topiq()(image_path))
        
        return QualityScore(
            image_path=image_path,
            clip_iqa=clip_score,
            topiq=topiq_score,
        )
    
    def compute_degradation_index(
        self,
        image_sequence: List[str]
    ) -> float:
        """
        Compute quality degradation across iteration sequence.
        Returns: Negative value indicates degradation.
        """
        if len(image_sequence) < 2:
            return 0.0
        
        scores = [self.assess(p).clip_iqa for p in image_sequence]
        # Compute slope of quality over iterations
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        return float(slope)
```

#### 2.4 Efficiency Metrics (`src/metrics/efficiency.py`)

```python
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
    correction_improvement_rate: float  # (final_success - initial_success) / initial_failures
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
        mean_iterations_to_success=np.mean([r.iterations_used for r in successes]) if successes else 0,
        mean_iterations_all=np.mean([r.iterations_used for r in results]) if results else 0,
        first_pass_success_rate=len(initial_correct) / len(results) if results else 0,
        correction_improvement_rate=len(corrected) / len(initial_wrong) if initial_wrong else 0,
        mean_time_seconds=np.mean([r.total_time_seconds for r in results]) if results else 0,
        mean_cost_usd=np.mean([r.cost_usd for r in results if r.cost_usd]) if any(r.cost_usd for r in results) else None,
    )

def correction_efficiency_score(results: List[LoopResult]) -> float:
    """
    Compute CES: Success weighted by iteration efficiency.
    CES = mean(success * (max_iter - iter_used + 1) / max_iter)
    Higher is better. Max = 1.0 (all succeed on first try)
    """
    scores = []
    for r in results:
        if r.success:
            efficiency = (r.max_iterations - r.iterations_used + 1) / r.max_iterations
        else:
            efficiency = 0
        scores.append(efficiency)
    return float(np.mean(scores))
```

#### 2.5 Agreement Metrics (`src/metrics/agreement.py`)

```python
"""Inter-rater agreement metrics for VLM evaluation."""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy import stats

@dataclass
class AgreementResult:
    """Agreement between VLM predictions and ground truth."""
    vlm_name: str
    accuracy: float              # Exact match rate
    pearson_correlation: float   # Correlation coefficient
    spearmans_rho: float         # Rank correlation
    cohens_kappa: float          # Agreement adjusted for chance
    mean_absolute_error: float

@dataclass
class VLMPrediction:
    """Single VLM prediction for an image."""
    image_id: str
    vlm_name: str
    predicted_count: int
    confidence: Optional[float] = None

@dataclass
class HumanAnnotation:
    """Human ground truth for an image."""
    image_id: str
    true_count: int
    annotator_counts: List[int]  # Multiple annotators

def compute_vlm_agreement(
    predictions: List[VLMPrediction],
    ground_truth: Dict[str, HumanAnnotation]
) -> Dict[str, AgreementResult]:
    """Compute agreement metrics for each VLM against human ground truth."""
    from collections import defaultdict
    from sklearn.metrics import cohen_kappa_score
    
    vlm_preds = defaultdict(list)
    vlm_truths = defaultdict(list)
    
    for pred in predictions:
        if pred.image_id in ground_truth:
            vlm_preds[pred.vlm_name].append(pred.predicted_count)
            vlm_truths[pred.vlm_name].append(ground_truth[pred.image_id].true_count)
    
    results = {}
    for vlm_name in vlm_preds:
        preds = np.array(vlm_preds[vlm_name])
        truths = np.array(vlm_truths[vlm_name])
        
        # Clip for kappa computation (needs bounded classes)
        preds_clipped = np.clip(preds, 0, 10)
        truths_clipped = np.clip(truths, 0, 10)
        
        results[vlm_name] = AgreementResult(
            vlm_name=vlm_name,
            accuracy=float(np.mean(preds == truths)),
            pearson_correlation=float(stats.pearsonr(preds, truths)[0]),
            spearmans_rho=float(stats.spearmanr(preds, truths)[0]),
            cohens_kappa=float(cohen_kappa_score(truths_clipped, preds_clipped)),
            mean_absolute_error=float(np.mean(np.abs(preds - truths))),
        )
    
    return results

def inter_vlm_agreement(
    predictions: List[VLMPrediction]
) -> float:
    """Compute Fleiss' kappa for agreement among multiple VLMs."""
    from collections import defaultdict
    from statsmodels.stats.inter_rater import fleiss_kappa
    
    # Group predictions by image
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred.image_id].append(pred.predicted_count)
    
    # Build rating matrix (images × count categories)
    max_count = 10
    n_images = len(by_image)
    rating_matrix = np.zeros((n_images, max_count + 1))
    
    for i, (img_id, counts) in enumerate(by_image.items()):
        for count in counts:
            if 0 <= count <= max_count:
                rating_matrix[i, count] += 1
    
    return float(fleiss_kappa(rating_matrix))
```

---

### 3. Enhanced Analyzer Module

Upgrade the analyzer module for structured output and multi-VLM support.

#### 3.1 File Structure

```
src/analyzers/
├── __init__.py
├── base.py              # Abstract base class
├── gemini.py            # Gemini 3 Pro analyzer
├── openai.py            # GPT-5.1 analyzer
├── anthropic.py         # Claude 4.5 Opus analyzer
├── openrouter.py        # OpenRouter models (Qwen, Llama)
├── ensemble.py          # Multi-VLM ensemble
└── prompts.py           # Counting prompt templates
```

#### 3.2 Base Analyzer (`src/analyzers/base.py`)

```python
"""Base class for VLM analyzers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import base64

@dataclass
class BoundingBox:
    """Object location in image."""
    x_min: float  # 0-1 normalized
    y_min: float
    x_max: float
    y_max: float
    confidence: Optional[float] = None

@dataclass
class CountAnalysisResult:
    """Structured counting analysis result."""
    count: int
    confidence: float  # 0-1
    object_type: str
    reasoning: str
    potential_ambiguities: List[str]
    object_locations: Optional[List[BoundingBox]] = None
    raw_response: Optional[str] = None

class BaseAnalyzer(ABC):
    """Abstract base class for image analyzers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def analyze(
        self,
        image_path: str,
        object_type: str,
        target_count: Optional[int] = None
    ) -> CountAnalysisResult:
        """
        Analyze image and count objects.
        
        Args:
            image_path: Path to image file
            object_type: Type of object to count (e.g., "apples")
            target_count: Expected count (for verification mode)
        
        Returns:
            Structured analysis result
        """
        pass
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", 
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        return mime_types.get(suffix, "image/jpeg")
```

#### 3.3 Counting Prompts (`src/analyzers/prompts.py`)

```python
"""Prompt templates for counting analysis."""

COUNTING_SYSTEM_PROMPT = """You are an expert image analyst specializing in precise object counting. 
Your task is to count specific objects in images with high accuracy.

Guidelines:
1. Count carefully and methodically
2. Consider partially visible objects (count if >50% visible)
3. Distinguish the target object from similar objects
4. Note any ambiguities or uncertainties
5. Provide your confidence level (0-1)

Respond in the following JSON format:
{
    "count": <integer>,
    "confidence": <float 0-1>,
    "reasoning": "<step-by-step counting explanation>",
    "ambiguities": ["<potential issue 1>", "<potential issue 2>"],
    "locations": [
        {"description": "<location of object 1>"},
        {"description": "<location of object 2>"}
    ]
}"""

def get_counting_prompt(object_type: str, target_count: Optional[int] = None) -> str:
    """Generate counting prompt for specific object type."""
    base = f"Count the exact number of {object_type} in this image."
    
    if target_count is not None:
        base += f"\n\nNote: The expected count is {target_count}. Verify if this is correct."
    
    return base

def get_verification_prompt(object_type: str, claimed_count: int) -> str:
    """Generate verification prompt."""
    return f"""Verify the count of {object_type} in this image.
    
A previous analysis claimed there are {claimed_count} {object_type}.

Is this count correct? If not, what is the actual count?

Respond in JSON format:
{{
    "claimed_count": {claimed_count},
    "actual_count": <integer>,
    "is_correct": <boolean>,
    "confidence": <float 0-1>,
    "reasoning": "<explanation>"
}}"""

def get_correction_feedback_prompt(
    object_type: str,
    target_count: int,
    current_count: int
) -> str:
    """Generate feedback for image correction."""
    diff = target_count - current_count
    
    if diff > 0:
        action = f"add {diff} more"
    elif diff < 0:
        action = f"remove {abs(diff)}"
    else:
        return "The count is correct. No changes needed."
    
    return f"""The image currently has {current_count} {object_type}, but should have {target_count}.

Please {action} {object_type} to reach the target count of {target_count}.

Provide specific guidance:
1. Where should objects be added/removed?
2. What should the corrected image look like?
3. Any specific placement recommendations?"""
```

#### 3.4 Claude Analyzer (`src/analyzers/anthropic.py`)

```python
"""Claude 4.5 Opus analyzer implementation."""

import json
import anthropic
from typing import Optional
from .base import BaseAnalyzer, CountAnalysisResult
from .prompts import COUNTING_SYSTEM_PROMPT, get_counting_prompt

class ClaudeAnalyzer(BaseAnalyzer):
    """Analyzer using Claude 4.5 Opus."""
    
    def __init__(self):
        super().__init__("claude-opus-4-5")
        self.client = anthropic.Anthropic()
    
    async def analyze(
        self,
        image_path: str,
        object_type: str,
        target_count: Optional[int] = None
    ) -> CountAnalysisResult:
        """Analyze image using Claude 4.5 Opus."""
        
        image_data = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        
        message = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=COUNTING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": get_counting_prompt(object_type, target_count),
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            data = json.loads(json_str.strip())
            
            return CountAnalysisResult(
                count=data["count"],
                confidence=data.get("confidence", 0.8),
                object_type=object_type,
                reasoning=data.get("reasoning", ""),
                potential_ambiguities=data.get("ambiguities", []),
                raw_response=response_text,
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: try to extract count from text
            import re
            numbers = re.findall(r'\b(\d+)\b', response_text)
            count = int(numbers[0]) if numbers else 0
            
            return CountAnalysisResult(
                count=count,
                confidence=0.5,
                object_type=object_type,
                reasoning=f"Parsed from unstructured response: {response_text[:200]}",
                potential_ambiguities=["Response was not in expected JSON format"],
                raw_response=response_text,
            )
```

#### 3.5 GPT-5.1 Analyzer (`src/analyzers/openai.py`)

```python
"""GPT-5.1 analyzer implementation."""

import json
from openai import OpenAI
from typing import Optional
from .base import BaseAnalyzer, CountAnalysisResult
from .prompts import COUNTING_SYSTEM_PROMPT, get_counting_prompt

class GPT5Analyzer(BaseAnalyzer):
    """Analyzer using GPT-5.1."""
    
    def __init__(self):
        super().__init__("gpt-5.1")
        self.client = OpenAI()
    
    async def analyze(
        self,
        image_path: str,
        object_type: str,
        target_count: Optional[int] = None
    ) -> CountAnalysisResult:
        """Analyze image using GPT-5.1."""
        
        image_data = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        
        response = self.client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {
                    "role": "system",
                    "content": COUNTING_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": get_counting_prompt(object_type, target_count)
                        }
                    ]
                }
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        data = json.loads(response_text)
        
        return CountAnalysisResult(
            count=data["count"],
            confidence=data.get("confidence", 0.8),
            object_type=object_type,
            reasoning=data.get("reasoning", ""),
            potential_ambiguities=data.get("ambiguities", []),
            raw_response=response_text,
        )
```

#### 3.6 Ensemble Analyzer (`src/analyzers/ensemble.py`)

```python
"""Multi-VLM ensemble analyzer."""

from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import Counter
import numpy as np

from .base import BaseAnalyzer, CountAnalysisResult
from .anthropic import ClaudeAnalyzer
from .openai import GPT5Analyzer
from .gemini import GeminiAnalyzer
from .openrouter import QwenAnalyzer

@dataclass
class EnsembleResult:
    """Result from ensemble analysis."""
    consensus_count: int
    confidence: float
    individual_results: Dict[str, CountAnalysisResult]
    agreement_rate: float  # Fraction agreeing with consensus
    voting_method: str

class EnsembleAnalyzer:
    """Ensemble of multiple VLM analyzers."""
    
    def __init__(
        self,
        analyzers: Optional[List[str]] = None,
        voting_method: str = "majority"  # "majority", "confidence_weighted", "median"
    ):
        self.voting_method = voting_method
        
        # Default to all available analyzers
        analyzer_classes = {
            "claude": ClaudeAnalyzer,
            "gpt5": GPT5Analyzer,
            "gemini": GeminiAnalyzer,
            "qwen": QwenAnalyzer,
        }
        
        if analyzers is None:
            analyzers = list(analyzer_classes.keys())
        
        self.analyzers = {
            name: analyzer_classes[name]()
            for name in analyzers
            if name in analyzer_classes
        }
    
    async def analyze(
        self,
        image_path: str,
        object_type: str,
        target_count: Optional[int] = None
    ) -> EnsembleResult:
        """Run all analyzers and compute consensus."""
        import asyncio
        
        # Run all analyzers concurrently
        tasks = {
            name: analyzer.analyze(image_path, object_type, target_count)
            for name, analyzer in self.analyzers.items()
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                print(f"Analyzer {name} failed: {e}")
        
        # Compute consensus
        counts = [r.count for r in results.values()]
        confidences = [r.confidence for r in results.values()]
        
        if self.voting_method == "majority":
            consensus = Counter(counts).most_common(1)[0][0]
        elif self.voting_method == "confidence_weighted":
            # Weighted average rounded to nearest integer
            weighted_sum = sum(c * conf for c, conf in zip(counts, confidences))
            total_conf = sum(confidences)
            consensus = round(weighted_sum / total_conf) if total_conf > 0 else counts[0]
        elif self.voting_method == "median":
            consensus = int(np.median(counts))
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        agreement_rate = sum(1 for c in counts if c == consensus) / len(counts)
        
        # Confidence based on agreement
        ensemble_confidence = agreement_rate * np.mean(confidences)
        
        return EnsembleResult(
            consensus_count=consensus,
            confidence=ensemble_confidence,
            individual_results=results,
            agreement_rate=agreement_rate,
            voting_method=self.voting_method,
        )
```

---

### 4. Enhanced Editor Module

Add support for targeted corrections and new models.

#### 4.1 File Structure

```
src/editors/
├── __init__.py
├── base.py              # Abstract base class
├── openai.py            # GPT Image 1 editor
├── flux.py              # FLUX.2 Pro Edit editor
├── strategies.py        # Correction strategies
└── prompt_refinement.py # LLM-powered prompt improvement
```

#### 4.2 Correction Strategies (`src/editors/strategies.py`)

```python
"""Correction strategies for iterative editing."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

class CorrectionStrategy(Enum):
    FULL_REGENERATION = "full_regeneration"
    INPAINTING = "inpainting"
    PROMPT_REFINEMENT = "prompt_refinement"
    HYBRID = "hybrid"

@dataclass
class CorrectionPlan:
    """Plan for correcting an image."""
    strategy: CorrectionStrategy
    refined_prompt: Optional[str] = None
    mask_description: Optional[str] = None
    target_region: Optional[str] = None  # "add to bottom-left", "remove from center"
    edit_strength: float = 0.8

class CorrectionPlanner:
    """Plans correction strategy based on error analysis."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def plan_correction(
        self,
        original_prompt: str,
        target_count: int,
        current_count: int,
        vlm_reasoning: str,
        iteration: int,
        max_iterations: int
    ) -> CorrectionPlan:
        """
        Determine best correction strategy.
        
        Heuristics:
        - Small error (±1-2): Try inpainting first
        - Large error: Full regeneration with refined prompt
        - Late iterations: More aggressive refinement
        """
        error = abs(target_count - current_count)
        
        # Simple heuristic-based planning
        if error <= 2 and iteration < max_iterations - 1:
            strategy = CorrectionStrategy.INPAINTING
        elif iteration >= max_iterations - 1:
            strategy = CorrectionStrategy.HYBRID
        else:
            strategy = CorrectionStrategy.PROMPT_REFINEMENT
        
        # Get refined prompt from LLM
        refined_prompt = await self._refine_prompt(
            original_prompt, target_count, current_count, vlm_reasoning
        )
        
        return CorrectionPlan(
            strategy=strategy,
            refined_prompt=refined_prompt,
            edit_strength=0.7 + (0.1 * iteration),  # Increase strength each iteration
        )
    
    async def _refine_prompt(
        self,
        original: str,
        target: int,
        current: int,
        reasoning: str
    ) -> str:
        """Use LLM to refine prompt for better counting."""
        
        system = """You are an expert at writing prompts for image generation models.
Your task is to refine prompts to achieve exact object counts.

Guidelines:
1. Be explicit about the exact count ("exactly 5", "precisely 3")
2. Use spatial arrangements ("arranged in a row", "evenly spaced")
3. Avoid ambiguous language
4. Keep the original style/context intact"""

        user = f"""Original prompt: "{original}"
Target count: {target}
Current result: {current} objects
Analysis: {reasoning}

Rewrite the prompt to achieve exactly {target} objects. Only output the refined prompt, nothing else."""

        # This would call the LLM - implementation depends on which LLM client is used
        # For now, return a simple refinement
        diff = target - current
        emphasis = "exactly" if abs(diff) <= 2 else "precisely"
        
        # Extract object type from original prompt (simplified)
        import re
        numbers = re.findall(r'\d+', original)
        if numbers:
            refined = re.sub(r'\d+', f'{emphasis} {target}', original, count=1)
        else:
            refined = f"{emphasis} {target} " + original
        
        return refined
```

#### 4.3 Prompt Refinement (`src/editors/prompt_refinement.py`)

```python
"""LLM-powered prompt refinement for better counting accuracy."""

from dataclasses import dataclass
from typing import List, Optional
import anthropic

@dataclass
class RefinedPrompt:
    """Refined prompt with metadata."""
    original: str
    refined: str
    changes_made: List[str]
    confidence: float

class PromptRefiner:
    """Refines prompts using Claude for better counting accuracy."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    async def refine_for_counting(
        self,
        prompt: str,
        target_count: int,
        object_type: str,
        previous_attempts: Optional[List[dict]] = None
    ) -> RefinedPrompt:
        """
        Refine prompt to improve counting accuracy.
        
        Args:
            prompt: Original generation prompt
            target_count: Desired object count
            object_type: Type of object being counted
            previous_attempts: List of {prompt, result_count} from failed attempts
        """
        
        history = ""
        if previous_attempts:
            history = "\n\nPrevious failed attempts:\n"
            for i, attempt in enumerate(previous_attempts, 1):
                history += f"{i}. Prompt: \"{attempt['prompt']}\" → Got {attempt['count']} (wanted {target_count})\n"
        
        message = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"""Rewrite this image generation prompt to ensure exactly {target_count} {object_type} appear in the generated image.

Original prompt: "{prompt}"
Target: exactly {target_count} {object_type}
{history}

Techniques to use:
1. Explicit count emphasis ("exactly N", "precisely N", "N and only N")
2. Spatial arrangement ("N items arranged in a row/grid/circle")
3. Numbered references ("first, second, third...")
4. Negative constraints ("no more than N", "not {target_count + 1}")

Output only the refined prompt, nothing else."""
                }
            ],
        )
        
        refined = message.content[0].text.strip().strip('"')
        
        # Analyze changes
        changes = []
        if "exactly" in refined.lower() and "exactly" not in prompt.lower():
            changes.append("Added explicit count emphasis")
        if "arranged" in refined.lower():
            changes.append("Added spatial arrangement")
        if any(word in refined.lower() for word in ["first", "second", "third"]):
            changes.append("Added numbered references")
        
        return RefinedPrompt(
            original=prompt,
            refined=refined,
            changes_made=changes,
            confidence=0.8 if len(changes) > 0 else 0.6,
        )
```

---

### 5. Benchmark Dataset

Create a structured benchmark for systematic evaluation.

#### 5.1 File Structure

```
src/benchmark/
├── __init__.py
├── prompts.py           # Prompt generation
├── dataset.py           # Dataset management
└── categories.py        # Object categories
```

#### 5.2 Prompt Categories (`src/benchmark/categories.py`)

```python
"""Object categories for benchmark prompts."""

OBJECT_CATEGORIES = {
    # Simple, countable objects
    "simple": [
        "apples",
        "oranges",
        "bananas",
        "lemons",
        "strawberries",
        "eggs",
        "cupcakes",
        "donuts",
        "cookies",
        "candles",
    ],
    
    # Animals
    "animals": [
        "cats",
        "dogs",
        "birds",
        "fish",
        "butterflies",
        "rabbits",
        "ducks",
        "chickens",
        "frogs",
        "turtles",
    ],
    
    # Objects with occlusion potential
    "occludable": [
        "books",
        "boxes",
        "cups",
        "plates",
        "chairs",
        "bottles",
        "vases",
        "jars",
        "bowls",
        "baskets",
    ],
    
    # Small objects (harder to count)
    "small": [
        "marbles",
        "coins",
        "buttons",
        "beads",
        "cherries",
        "grapes",
        "peas",
        "nuts",
        "pills",
        "seeds",
    ],
    
    # People/characters
    "people": [
        "people",
        "children",
        "dancers",
        "runners",
        "musicians",
        "chefs",
        "artists",
        "students",
        "workers",
        "athletes",
    ],
    
    # Abstract/challenging
    "challenging": [
        "shadows",
        "reflections",
        "clouds",
        "stars",
        "bubbles",
        "raindrops",
        "snowflakes",
        "leaves",
        "petals",
        "waves",
    ],
}

SCENE_TEMPLATES = [
    "{count} {object} on a wooden table",
    "{count} {object} in a basket",
    "{count} {object} arranged in a row",
    "{count} {object} scattered on the floor",
    "{count} {object} on a white background",
    "{count} {object} in a garden",
    "{count} {object} on a kitchen counter",
    "{count} {object} in a living room",
    "{count} {object} floating in water",
    "{count} {object} on a checkered tablecloth",
]

COUNT_RANGE = list(range(1, 11))  # 1-10
```

#### 5.3 Dataset Generator (`src/benchmark/dataset.py`)

```python
"""Benchmark dataset generation and management."""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from pathlib import Path
import json
import random
from datetime import datetime

from .categories import OBJECT_CATEGORIES, SCENE_TEMPLATES, COUNT_RANGE

@dataclass
class BenchmarkPrompt:
    """Single benchmark prompt."""
    id: str
    prompt: str
    target_count: int
    object_type: str
    category: str
    scene_template: str
    
@dataclass
class BenchmarkDataset:
    """Complete benchmark dataset."""
    name: str
    version: str
    created_at: str
    prompts: List[BenchmarkPrompt]
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __iter__(self) -> Iterator[BenchmarkPrompt]:
        return iter(self.prompts)
    
    def filter_by_category(self, category: str) -> List[BenchmarkPrompt]:
        return [p for p in self.prompts if p.category == category]
    
    def filter_by_count(self, count: int) -> List[BenchmarkPrompt]:
        return [p for p in self.prompts if p.target_count == count]
    
    def save(self, path: str):
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "prompts": [
                {
                    "id": p.id,
                    "prompt": p.prompt,
                    "target_count": p.target_count,
                    "object_type": p.object_type,
                    "category": p.category,
                    "scene_template": p.scene_template,
                }
                for p in self.prompts
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        prompts = [
            BenchmarkPrompt(**p) for p in data["prompts"]
        ]
        
        return cls(
            name=data["name"],
            version=data["version"],
            created_at=data["created_at"],
            prompts=prompts,
            metadata=data.get("metadata", {}),
        )

def generate_benchmark_dataset(
    name: str = "imgcount-benchmark",
    version: str = "1.0",
    samples_per_combination: int = 1,
    categories: Optional[List[str]] = None,
    counts: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> BenchmarkDataset:
    """
    Generate a systematic benchmark dataset.
    
    Args:
        name: Dataset name
        version: Version string
        samples_per_combination: Number of prompts per (object, count) pair
        categories: Which categories to include (default: all)
        counts: Which counts to include (default: 1-10)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    if categories is None:
        categories = list(OBJECT_CATEGORIES.keys())
    
    if counts is None:
        counts = COUNT_RANGE
    
    prompts = []
    prompt_id = 0
    
    for category in categories:
        objects = OBJECT_CATEGORIES[category]
        
        for obj in objects:
            for count in counts:
                for _ in range(samples_per_combination):
                    template = random.choice(SCENE_TEMPLATES)
                    prompt_text = template.format(count=count, object=obj)
                    
                    prompts.append(BenchmarkPrompt(
                        id=f"{name}-{prompt_id:05d}",
                        prompt=prompt_text,
                        target_count=count,
                        object_type=obj,
                        category=category,
                        scene_template=template,
                    ))
                    prompt_id += 1
    
    return BenchmarkDataset(
        name=name,
        version=version,
        created_at=datetime.utcnow().isoformat(),
        prompts=prompts,
        metadata={
            "categories": categories,
            "counts": counts,
            "samples_per_combination": samples_per_combination,
            "total_prompts": len(prompts),
            "seed": seed,
        },
    )
```

---

### 6. Experiment Runner

Create a comprehensive experiment execution framework.

#### 6.1 File Structure

```
src/experiments/
├── __init__.py
├── runner.py            # Main experiment runner
├── configs.py           # Experiment configurations
├── logging.py           # Structured logging
└── analysis.py          # Result analysis
```

#### 6.2 Experiment Configuration (`src/experiments/configs.py`)

```python
"""Experiment configurations."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class ExperimentType(Enum):
    BASELINE = "baseline"           # Single-pass generation
    VLM_RELIABILITY = "vlm_reliability"  # VLM accuracy evaluation
    CORRECTION_LOOP = "correction_loop"  # Iterative correction
    ABLATION = "ablation"           # Ablation studies

@dataclass
class BaselineConfig:
    """Configuration for baseline experiments."""
    generators: List[str]
    analyzers: List[str]
    dataset_path: str
    samples_per_prompt: int = 5
    output_dir: str = "results/baseline"

@dataclass
class CorrectionLoopConfig:
    """Configuration for correction loop experiments."""
    generator: str
    editor: str
    analyzer: str
    max_iterations: int = 3
    dataset_path: str = ""
    output_dir: str = "results/correction_loop"
    correction_strategy: str = "prompt_refinement"

@dataclass
class VLMReliabilityConfig:
    """Configuration for VLM reliability experiments."""
    analyzers: List[str]
    ground_truth_path: str  # Path to human annotations
    images_dir: str
    output_dir: str = "results/vlm_reliability"

@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    experiment_name: str
    base_config: CorrectionLoopConfig
    ablation_variable: str  # What to vary
    ablation_values: List[Any]
    output_dir: str = "results/ablation"

@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    name: str
    experiment_type: ExperimentType
    baseline: Optional[BaselineConfig] = None
    correction_loop: Optional[CorrectionLoopConfig] = None
    vlm_reliability: Optional[VLMReliabilityConfig] = None
    ablation: Optional[AblationConfig] = None
    
    # Common settings
    seed: int = 42
    save_images: bool = True
    save_intermediate: bool = True
    log_level: str = "INFO"
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "imgcount"
    wandb_entity: Optional[str] = None

# Predefined experiment configs
EXPERIMENT_PRESETS: Dict[str, ExperimentConfig] = {
    "baseline_all_models": ExperimentConfig(
        name="baseline_all_models",
        experiment_type=ExperimentType.BASELINE,
        baseline=BaselineConfig(
            generators=["gemini", "openai", "flux", "ideogram", "sd35"],
            analyzers=["claude", "gpt5", "gemini-vlm", "qwen"],
            dataset_path="data/benchmark_v1.json",
            samples_per_prompt=5,
        ),
    ),
    
    "vlm_reliability": ExperimentConfig(
        name="vlm_reliability",
        experiment_type=ExperimentType.VLM_RELIABILITY,
        vlm_reliability=VLMReliabilityConfig(
            analyzers=["claude", "gpt5", "gemini-vlm", "qwen"],
            ground_truth_path="data/human_annotations.json",
            images_dir="data/generated_images",
        ),
    ),
    
    "correction_loop_openai": ExperimentConfig(
        name="correction_loop_openai",
        experiment_type=ExperimentType.CORRECTION_LOOP,
        correction_loop=CorrectionLoopConfig(
            generator="openai",
            editor="openai",
            analyzer="claude",
            max_iterations=3,
            dataset_path="data/benchmark_v1.json",
        ),
    ),
    
    "ablation_max_iterations": ExperimentConfig(
        name="ablation_max_iterations",
        experiment_type=ExperimentType.ABLATION,
        ablation=AblationConfig(
            experiment_name="max_iterations",
            base_config=CorrectionLoopConfig(
                generator="openai",
                editor="openai", 
                analyzer="claude",
                max_iterations=1,
                dataset_path="data/benchmark_v1.json",
            ),
            ablation_variable="max_iterations",
            ablation_values=[1, 2, 3, 5],
        ),
    ),
}
```

#### 6.3 Experiment Runner (`src/experiments/runner.py`)

```python
"""Main experiment execution framework."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import logging

from .configs import ExperimentConfig, ExperimentType
from ..benchmark.dataset import BenchmarkDataset, BenchmarkPrompt
from ..metrics.counting import CountingResult, compute_counting_metrics
from ..metrics.efficiency import LoopResult, compute_efficiency_metrics
from ..models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs experiments based on configuration."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.output_dir = Path(self._get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.name,
                config=asdict(config),
            )
    
    def _get_output_dir(self) -> str:
        """Get output directory based on experiment type."""
        if self.config.experiment_type == ExperimentType.BASELINE:
            return self.config.baseline.output_dir
        elif self.config.experiment_type == ExperimentType.CORRECTION_LOOP:
            return self.config.correction_loop.output_dir
        elif self.config.experiment_type == ExperimentType.VLM_RELIABILITY:
            return self.config.vlm_reliability.output_dir
        elif self.config.experiment_type == ExperimentType.ABLATION:
            return self.config.ablation.output_dir
        return "results"
    
    async def run(self) -> Dict[str, Any]:
        """Execute the experiment."""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting experiment: {self.config.name}")
        
        try:
            if self.config.experiment_type == ExperimentType.BASELINE:
                results = await self._run_baseline()
            elif self.config.experiment_type == ExperimentType.CORRECTION_LOOP:
                results = await self._run_correction_loop()
            elif self.config.experiment_type == ExperimentType.VLM_RELIABILITY:
                results = await self._run_vlm_reliability()
            elif self.config.experiment_type == ExperimentType.ABLATION:
                results = await self._run_ablation()
            else:
                raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")
            
            # Save results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    async def _run_baseline(self) -> Dict[str, Any]:
        """Run baseline single-pass evaluation."""
        cfg = self.config.baseline
        dataset = BenchmarkDataset.load(cfg.dataset_path)
        
        all_results: List[CountingResult] = []
        
        for generator_name in cfg.generators:
            generator = self._load_generator(generator_name)
            
            for analyzer_name in cfg.analyzers:
                analyzer = self._load_analyzer(analyzer_name)
                
                for prompt in dataset:
                    for sample_idx in range(cfg.samples_per_prompt):
                        logger.info(f"Running {generator_name}/{analyzer_name}: {prompt.id} (sample {sample_idx})")
                        
                        # Generate image
                        image_path = await generator.generate(
                            prompt=prompt.prompt,
                            output_dir=str(self.output_dir / "images"),
                        )
                        
                        # Analyze
                        analysis = await analyzer.analyze(
                            image_path=image_path,
                            object_type=prompt.object_type,
                            target_count=prompt.target_count,
                        )
                        
                        result = CountingResult(
                            prompt=prompt.prompt,
                            target_count=prompt.target_count,
                            predicted_count=analysis.count,
                            object_type=prompt.object_type,
                            iteration=0,
                            generator=generator_name,
                            analyzer=analyzer_name,
                            image_path=image_path,
                            vlm_confidence=analysis.confidence,
                            vlm_reasoning=analysis.reasoning,
                        )
                        all_results.append(result)
        
        # Compute metrics
        overall_metrics = compute_counting_metrics(all_results)
        by_generator = {
            g: compute_counting_metrics([r for r in all_results if r.generator == g])
            for g in cfg.generators
        }
        by_analyzer = {
            a: compute_counting_metrics([r for r in all_results if r.analyzer == a])
            for a in cfg.analyzers
        }
        by_count = {
            c: compute_counting_metrics([r for r in all_results if r.target_count == c])
            for c in range(1, 11)
        }
        
        return {
            "experiment": self.config.name,
            "type": "baseline",
            "overall_metrics": asdict(overall_metrics),
            "by_generator": {k: asdict(v) for k, v in by_generator.items()},
            "by_analyzer": {k: asdict(v) for k, v in by_analyzer.items()},
            "by_count": {k: asdict(v) for k, v in by_count.items()},
            "raw_results": [asdict(r) for r in all_results],
        }
    
    async def _run_correction_loop(self) -> Dict[str, Any]:
        """Run correction loop evaluation."""
        cfg = self.config.correction_loop
        dataset = BenchmarkDataset.load(cfg.dataset_path)
        
        generator = self._load_generator(cfg.generator)
        editor = self._load_editor(cfg.editor)
        analyzer = self._load_analyzer(cfg.analyzer)
        
        loop_results: List[LoopResult] = []
        all_counting_results: List[CountingResult] = []
        
        for prompt in dataset:
            logger.info(f"Processing: {prompt.id}")
            
            start_time = datetime.utcnow()
            
            # Initial generation
            image_path = await generator.generate(
                prompt=prompt.prompt,
                output_dir=str(self.output_dir / "images"),
            )
            
            analysis = await analyzer.analyze(
                image_path=image_path,
                object_type=prompt.object_type,
                target_count=prompt.target_count,
            )
            
            initial_count = analysis.count
            current_count = initial_count
            current_image = image_path
            iterations_used = 0
            
            # Store initial result
            all_counting_results.append(CountingResult(
                prompt=prompt.prompt,
                target_count=prompt.target_count,
                predicted_count=initial_count,
                object_type=prompt.object_type,
                iteration=0,
                generator=cfg.generator,
                analyzer=cfg.analyzer,
                image_path=image_path,
            ))
            
            # Correction loop
            while current_count != prompt.target_count and iterations_used < cfg.max_iterations:
                iterations_used += 1
                
                # Edit image
                edit_prompt = self._generate_edit_prompt(
                    prompt.prompt,
                    prompt.target_count,
                    current_count,
                    prompt.object_type,
                )
                
                current_image = await editor.edit(
                    image_path=current_image,
                    prompt=edit_prompt,
                    output_dir=str(self.output_dir / "images"),
                )
                
                # Re-analyze
                analysis = await analyzer.analyze(
                    image_path=current_image,
                    object_type=prompt.object_type,
                    target_count=prompt.target_count,
                )
                
                current_count = analysis.count
                
                # Store iteration result
                all_counting_results.append(CountingResult(
                    prompt=prompt.prompt,
                    target_count=prompt.target_count,
                    predicted_count=current_count,
                    object_type=prompt.object_type,
                    iteration=iterations_used,
                    generator=cfg.generator,
                    analyzer=cfg.analyzer,
                    image_path=current_image,
                ))
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            
            loop_results.append(LoopResult(
                prompt=prompt.prompt,
                target_count=prompt.target_count,
                initial_count=initial_count,
                final_count=current_count,
                iterations_used=iterations_used,
                max_iterations=cfg.max_iterations,
                success=(current_count == prompt.target_count),
                total_time_seconds=elapsed,
            ))
        
        # Compute metrics
        efficiency = compute_efficiency_metrics(loop_results)
        counting = compute_counting_metrics(all_counting_results)
        
        return {
            "experiment": self.config.name,
            "type": "correction_loop",
            "efficiency_metrics": asdict(efficiency),
            "counting_metrics": asdict(counting),
            "loop_results": [asdict(r) for r in loop_results],
            "counting_results": [asdict(r) for r in all_counting_results],
        }
    
    def _generate_edit_prompt(
        self,
        original_prompt: str,
        target: int,
        current: int,
        object_type: str,
    ) -> str:
        """Generate edit prompt for correction."""
        diff = target - current
        if diff > 0:
            return f"Add {diff} more {object_type} to the image. Target: exactly {target} {object_type}."
        else:
            return f"Remove {abs(diff)} {object_type} from the image. Target: exactly {target} {object_type}."
    
    def _load_generator(self, name: str):
        """Load generator by name."""
        # Import and instantiate generator
        from ..generators import get_generator
        return get_generator(name)
    
    def _load_editor(self, name: str):
        """Load editor by name."""
        from ..editors import get_editor
        return get_editor(name)
    
    def _load_analyzer(self, name: str):
        """Load analyzer by name."""
        from ..analyzers import get_analyzer
        return get_analyzer(name)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.config.name}_{timestamp}.json"
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            import wandb
            wandb.log(results.get("overall_metrics", results.get("efficiency_metrics", {})))
            wandb.save(str(output_path))
```

---

### 7. CLI Updates

Update `main.py` with new commands and options.

#### 7.1 Updated CLI (`main.py`)

```python
"""
Image Generation and Analysis Platform CLI.

Usage:
    uv run python main.py generate --prompt "5 apples" --count 5 --generator gemini --analyzer claude
    uv run python main.py benchmark --config baseline_all_models
    uv run python main.py analyze --image path/to/image.png --object apples --analyzer ensemble
"""

import argparse
import asyncio
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Image Generation and Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate command (original functionality)
    gen_parser = subparsers.add_parser("generate", help="Generate and analyze a single image")
    gen_parser.add_argument("--prompt", required=True, help="Generation prompt")
    gen_parser.add_argument("--count", type=int, required=True, help="Target object count")
    gen_parser.add_argument("--object", required=True, help="Object type to count")
    gen_parser.add_argument("--mode", choices=["direct", "loop"], default="direct")
    gen_parser.add_argument("--generator", default="gemini", 
                           choices=["gemini", "openai", "flux", "ideogram", "sd35", "recraft"])
    gen_parser.add_argument("--editor", default="openai",
                           choices=["openai", "flux-edit", "recraft-edit"])
    gen_parser.add_argument("--analyzer", default="claude",
                           choices=["claude", "gpt5", "gemini-vlm", "qwen", "ensemble"])
    gen_parser.add_argument("--max-iterations", type=int, default=3)
    gen_parser.add_argument("--output-dir", default="output")
    
    # Benchmark command (new)
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark experiments")
    bench_parser.add_argument("--config", required=True, help="Experiment config name or path")
    bench_parser.add_argument("--output-dir", default="results")
    bench_parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    bench_parser.add_argument("--seed", type=int, default=42)
    
    # Analyze command (new)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing image(s)")
    analyze_parser.add_argument("--image", required=True, help="Image path or directory")
    analyze_parser.add_argument("--object", required=True, help="Object type to count")
    analyze_parser.add_argument("--analyzer", default="ensemble",
                               choices=["claude", "gpt5", "gemini-vlm", "qwen", "ensemble"])
    analyze_parser.add_argument("--output", help="Output JSON path")
    
    # Dataset command (new)
    dataset_parser = subparsers.add_parser("dataset", help="Generate benchmark dataset")
    dataset_parser.add_argument("--output", required=True, help="Output JSON path")
    dataset_parser.add_argument("--categories", nargs="+", help="Object categories to include")
    dataset_parser.add_argument("--counts", nargs="+", type=int, help="Counts to include")
    dataset_parser.add_argument("--samples", type=int, default=1, help="Samples per combination")
    dataset_parser.add_argument("--seed", type=int, default=42)
    
    # Evaluate command (new)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results and compute metrics")
    eval_parser.add_argument("--results", required=True, help="Results JSON path")
    eval_parser.add_argument("--output", help="Output metrics path")
    eval_parser.add_argument("--format", choices=["json", "markdown", "latex"], default="json")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        asyncio.run(run_generate(args))
    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args))
    elif args.command == "analyze":
        asyncio.run(run_analyze(args))
    elif args.command == "dataset":
        run_dataset(args)
    elif args.command == "evaluate":
        run_evaluate(args)

async def run_generate(args):
    """Run single image generation."""
    from src.generators import get_generator
    from src.analyzers import get_analyzer
    from src.editors import get_editor
    
    generator = get_generator(args.generator)
    analyzer = get_analyzer(args.analyzer)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate initial image
    print(f"Generating image with {args.generator}...")
    image_path = await generator.generate(
        prompt=args.prompt,
        output_dir=str(output_dir),
    )
    print(f"Generated: {image_path}")
    
    # Analyze
    print(f"Analyzing with {args.analyzer}...")
    result = await analyzer.analyze(
        image_path=image_path,
        object_type=args.object,
        target_count=args.count,
    )
    
    print(f"Count: {result.count} (target: {args.count})")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    if args.mode == "loop" and result.count != args.count:
        editor = get_editor(args.editor)
        
        for iteration in range(1, args.max_iterations + 1):
            if result.count == args.count:
                break
            
            print(f"\nIteration {iteration}: Editing image...")
            
            diff = args.count - result.count
            edit_prompt = f"{'Add' if diff > 0 else 'Remove'} {abs(diff)} {args.object}. Target: exactly {args.count}."
            
            image_path = await editor.edit(
                image_path=image_path,
                prompt=edit_prompt,
                output_dir=str(output_dir),
            )
            
            result = await analyzer.analyze(
                image_path=image_path,
                object_type=args.object,
                target_count=args.count,
            )
            
            print(f"New count: {result.count}")
        
        if result.count == args.count:
            print("\n✓ Successfully achieved target count!")
        else:
            print(f"\n✗ Could not achieve target count after {args.max_iterations} iterations")

async def run_benchmark(args):
    """Run benchmark experiment."""
    from src.experiments.configs import EXPERIMENT_PRESETS, ExperimentConfig
    from src.experiments.runner import ExperimentRunner
    
    # Load config
    if args.config in EXPERIMENT_PRESETS:
        config = EXPERIMENT_PRESETS[args.config]
    else:
        # Try to load from file
        import json
        with open(args.config) as f:
            config = ExperimentConfig(**json.load(f))
    
    # Override settings
    config.seed = args.seed
    config.use_wandb = args.wandb
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = await runner.run()
    
    print(f"\nExperiment complete: {config.name}")
    if "overall_metrics" in results:
        print(f"Accuracy: {results['overall_metrics']['counting_accuracy_rate']:.2%}")
        print(f"MAE: {results['overall_metrics']['mean_absolute_error']:.2f}")

async def run_analyze(args):
    """Analyze existing images."""
    from src.analyzers import get_analyzer
    import json
    
    analyzer = get_analyzer(args.analyzer)
    
    image_path = Path(args.image)
    
    if image_path.is_dir():
        images = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))
    else:
        images = [image_path]
    
    results = []
    for img in images:
        print(f"Analyzing: {img}")
        result = await analyzer.analyze(
            image_path=str(img),
            object_type=args.object,
        )
        results.append({
            "image": str(img),
            "count": result.count,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        })
        print(f"  Count: {result.count}, Confidence: {result.confidence:.2f}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

def run_dataset(args):
    """Generate benchmark dataset."""
    from src.benchmark.dataset import generate_benchmark_dataset
    
    dataset = generate_benchmark_dataset(
        categories=args.categories,
        counts=args.counts,
        samples_per_combination=args.samples,
        seed=args.seed,
    )
    
    dataset.save(args.output)
    print(f"Generated dataset with {len(dataset)} prompts: {args.output}")

def run_evaluate(args):
    """Evaluate results and compute metrics."""
    import json
    from src.metrics.counting import CountingResult, compute_counting_metrics
    
    with open(args.results) as f:
        data = json.load(f)
    
    # Convert to CountingResult objects
    results = [CountingResult(**r) for r in data.get("raw_results", data.get("counting_results", []))]
    
    metrics = compute_counting_metrics(results)
    
    if args.format == "json":
        output = {
            "counting_accuracy_rate": metrics.counting_accuracy_rate,
            "mean_absolute_error": metrics.mean_absolute_error,
            "within_one_accuracy": metrics.within_one_accuracy,
            "n_samples": metrics.n_samples,
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(json.dumps(output, indent=2))
    
    elif args.format == "markdown":
        md = f"""# Evaluation Results

| Metric | Value |
|--------|-------|
| Counting Accuracy | {metrics.counting_accuracy_rate:.2%} |
| Mean Absolute Error | {metrics.mean_absolute_error:.2f} |
| Within ±1 Accuracy | {metrics.within_one_accuracy:.2%} |
| N Samples | {metrics.n_samples} |
"""
        if args.output:
            with open(args.output, "w") as f:
                f.write(md)
        else:
            print(md)

if __name__ == "__main__":
    main()
```

---

### 8. Dependencies Update

#### 8.1 Updated `pyproject.toml`

```toml
[project]
name = "imgcount"
version = "0.2.0"
description = "Image generation and analysis platform for evaluating generative numeracy"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"

dependencies = [
    # Core
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "aiofiles>=23.0.0",
    "pydantic>=2.0.0",
    
    # API Clients
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "google-generativeai>=0.8.0",
    "fal-client>=0.5.0",
    
    # Image Processing
    "pillow>=10.0.0",
    "numpy>=1.26.0",
    
    # Quality Assessment
    "pyiqa>=0.1.10",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    
    # Metrics
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "statsmodels>=0.14.0",
    
    # Experiment Tracking (optional)
    "wandb>=0.16.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

### 9. Environment Configuration

#### 9.1 Updated `.env.example`

```bash
# Google AI (Gemini)
GEMINI_API_KEY=your_gemini_api_key

# OpenAI (GPT-5.1, GPT Image 1)
OPENAI_API_KEY=your_openai_api_key

# Anthropic (Claude 4.5 Opus)
ANTHROPIC_API_KEY=your_anthropic_api_key

# OpenRouter (Qwen, Llama, etc.)
OPENROUTER_API_KEY=your_openrouter_api_key

# Fal.ai (FLUX.2, Recraft)
FAL_KEY=your_fal_api_key

# Optional: Ideogram
IDEOGRAM_API_KEY=your_ideogram_api_key

# Optional: Stability AI
STABILITY_API_KEY=your_stability_api_key

# Optional: Weights & Biases
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=imgcount
WANDB_ENTITY=your_entity

# Optional: Replicate
REPLICATE_API_TOKEN=your_replicate_token
```

---

### 10. Testing Requirements

#### 10.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_metrics/
│   ├── test_counting.py
│   ├── test_quality.py
│   └── test_agreement.py
├── test_analyzers/
│   ├── test_base.py
│   ├── test_ensemble.py
│   └── test_prompts.py
├── test_benchmark/
│   ├── test_dataset.py
│   └── test_categories.py
├── test_experiments/
│   ├── test_configs.py
│   └── test_runner.py
└── test_integration/
    └── test_full_pipeline.py
```

#### 10.2 Key Test Cases

```python
# tests/test_metrics/test_counting.py

import pytest
from src.metrics.counting import (
    CountingResult,
    compute_counting_metrics,
    confusion_matrix,
)

@pytest.fixture
def sample_results():
    return [
        CountingResult(prompt="3 apples", target_count=3, predicted_count=3, ...),
        CountingResult(prompt="5 oranges", target_count=5, predicted_count=4, ...),
        CountingResult(prompt="2 bananas", target_count=2, predicted_count=2, ...),
        CountingResult(prompt="7 lemons", target_count=7, predicted_count=9, ...),
    ]

def test_counting_accuracy_rate(sample_results):
    metrics = compute_counting_metrics(sample_results)
    assert metrics.counting_accuracy_rate == 0.5  # 2/4 correct

def test_mean_absolute_error(sample_results):
    metrics = compute_counting_metrics(sample_results)
    # |0| + |1| + |0| + |2| = 3, /4 = 0.75
    assert metrics.mean_absolute_error == 0.75

def test_confusion_matrix_shape(sample_results):
    matrix = confusion_matrix(sample_results, max_count=10)
    assert matrix.shape == (11, 11)
```

---

### 11. Implementation Priority

Execute enhancements in this order:

1. **Phase 1: Core Infrastructure** (Week 1)
   - Model registry (`src/models/registry.py`)
   - Base classes for analyzers and editors
   - Environment setup

2. **Phase 2: Metrics Module** (Week 1-2)
   - All metrics files
   - Unit tests for metrics

3. **Phase 3: Analyzer Implementations** (Week 2)
   - Claude, GPT-5.1, Gemini analyzers
   - Ensemble analyzer
   - Structured prompts

4. **Phase 4: Editor Enhancements** (Week 2-3)
   - Correction strategies
   - Prompt refinement
   - New model integrations

5. **Phase 5: Benchmark System** (Week 3)
   - Dataset generation
   - Experiment configs
   - Runner implementation

6. **Phase 6: CLI & Integration** (Week 3-4)
   - Updated CLI
   - Integration tests
   - Documentation

---

### 12. Success Criteria

The implementation is complete when:

- [ ] All models in registry are functional and tested
- [ ] Metrics compute correctly with >95% test coverage
- [ ] All 5 VLM analyzers produce valid CountAnalysisResult
- [ ] Ensemble analyzer achieves higher accuracy than best individual
- [ ] Correction loop improves accuracy over baseline by >15%
- [ ] Benchmark dataset generates 1000+ prompts across all categories
- [ ] Full experiment runs complete in <24 hours on standard hardware
- [ ] CLI supports all documented commands
- [ ] All tests pass
- [ ] Documentation is complete

---

## Appendix A: API Reference Links

- Anthropic Claude: https://docs.anthropic.com/en/api
- OpenAI: https://platform.openai.com/docs
- Google Gemini: https://ai.google.dev/docs
- Fal.ai: https://fal.ai/docs
- OpenRouter: https://openrouter.ai/docs

## Appendix B: Model Verification

Before implementation, verify current model strings by checking:

```bash
# Anthropic
curl https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY"

# OpenAI  
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"

# Google
# Check https://ai.google.dev/models for latest Gemini models

# Fal.ai
# Check https://fal.ai/models for available models
```

Update MODEL_REGISTRY accordingly if model strings have changed.
