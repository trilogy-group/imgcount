# Development Guide

## Adding New Models

To add a new model, implement the appropriate interface from `src/models.py`.

### Adding a Generator
1.  Create a new class inheriting from `ImageGenerator` in `src/generators.py` (or a new file).
2.  Implement the `generate` method.
3.  Update `main.py` to include the new generator in the argument parser choices.

### Adding an Editor
1.  Create a new class inheriting from `ImageEditor` in `src/editors.py`.
2.  Implement the `edit` method.
3.  Update `main.py` to include the new editor.

### Adding an Analyzer
1.  Create a new class inheriting from `ImageAnalyzer` in `src/analyzers.py`.
2.  Implement the `analyze` method.
3.  Update `main.py` to include the new analyzer.

## Testing

Run unit tests using `uv`:

```bash
uv run python -m unittest tests/test_models.py
```

Tests mock the API calls, so they verify the logic of the `EvaluationLoop` and the integration of the classes, but do not make actual network requests.
