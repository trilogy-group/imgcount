# Architecture

The platform is built on a modular architecture defined in `src/models.py`.

## Core Components

### ImageGenerator
Abstract base class for models that generate images from text prompts.
*   `generate(prompt: str) -> ImageResult`

**Implementations**:
*   `GeminiGenerator`: Uses `gemini-2.5-flash-image`.
*   `OpenAIGenerator`: Uses `gpt-image-1`.
*   `FalGenerator`: Uses `fal-ai/recraft-v3`.

### ImageEditor
Abstract base class for models that edit existing images based on a prompt.
*   `edit(image_path: str, prompt: str) -> ImageResult`

**Implementations**:
*   `OpenAIEditor`: Uses `gpt-image-1` (Edit).
*   `FalEditor`: Uses `fal-ai/recraft-v3` (Edit).
*   `GeminiEditor`: Placeholder for `gemini-2.5-flash-image` (if supported).

### ImageAnalyzer
Abstract base class for VLMs that analyze images and return an object count.
*   `analyze(image_path: str, prompt: str) -> int`

**Implementations**:
*   `QwenAnalyzer`: Uses `qwen/qwen3-vl-235b-a22b-instruct` via OpenRouter.
*   `GeminiAnalyzer`: Uses `gemini-3-pro`.

### EvaluationLoop
The orchestrator class in `src/evaluator.py`.
1.  Calls `generator.generate()`.
2.  Calls `analyzer.analyze()`.
3.  If `mode="loop"` and count mismatches:
    *   Enters a retry loop (default max 2).
    *   Calls `editor.edit()`.
    *   Calls `analyzer.analyze()` again.
4.  Returns a result dictionary with the final image path, detected count, and step history.
