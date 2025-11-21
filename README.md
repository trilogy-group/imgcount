# Image Generation and Analysis Platform

This project implements an evaluation platform for image generation models, focusing on their ability to generate a specific number of objects. It supports direct generation, two-pass generation (base + edit), and automated analysis using Vision Language Models (VLMs).

## Features

*   **Direct Generation**: Generate images using SOTA models (Gemini 2.5 Flash Image, GPT Image 1, Recraft V3).
*   **Automated Analysis**: Count objects in generated images using VLMs (Qwen3 VL, Gemini 3 Pro).
*   **Auto-Correction Loop**: Automatically attempt to fix incorrect counts by editing the image (GPT Image 1, Recraft V3).
*   **Modular Architecture**: Easily extensible interfaces for Generators, Editors, and Analyzers.

## Installation

This project uses `uv` for dependency management.

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory with your API keys:
    ```env
    GEMINI_API_KEY=your_gemini_key
    OPENAI_API_KEY=your_openai_key
    OPENROUTER_API_KEY=your_openrouter_key
    FAL_KEY=your_fal_key
    ```

## Usage

Run the evaluation CLI using `uv run`. Images are saved to the `output/` directory.

### Direct Generation Mode
Generate an image and analyze it once.
```bash
uv run python main.py --prompt "3 apples on a table" --count 3 --object "apples" --mode direct --generator gemini --analyzer qwen
```

### Loop Mode (Auto-Correction)
Generate an image, analyze it, and if the count is wrong, attempt to edit it (up to 2 retries).
```bash
uv run python main.py --prompt "5 cats" --count 5 --mode loop --generator openai --editor openai --analyzer qwen
```

## Supported Models

| Type | Model | CLI Argument |
| :--- | :--- | :--- |
| **Generator** | Gemini 2.5 Flash Image | `gemini` |
| | GPT Image 1 | `openai` |
| | Recraft V3 (via Fal) | `fal` |
| **Editor** | GPT Image 1 | `openai` |
| | Recraft V3 (via Fal) | `fal` |
| | *Gemini Editor* | *(Coming Soon)* |
| **Analyzer** | Qwen3 VL 235B (OpenRouter) | `qwen` |
| | Gemini 3 Pro | `gemini` |

## Documentation

See [docs/](docs/) for more detailed documentation.
