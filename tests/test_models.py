import unittest
from unittest.mock import MagicMock, patch
from src.models import ImageResult
from src.generators import GeminiGenerator
from src.analyzers import QwenAnalyzer
from src.evaluator import EvaluationLoop

class TestEvaluationLoop(unittest.TestCase):
    def setUp(self):
        self.mock_generator = MagicMock()
        self.mock_analyzer = MagicMock()
        self.mock_editor = MagicMock()
        
        self.mock_generator.generate.return_value = ImageResult(image_path="test.png", metadata={})
        self.mock_analyzer.analyze.return_value = 3
        self.mock_editor.edit.return_value = ImageResult(image_path="edited.png", metadata={})

    def test_direct_mode_match(self):
        loop = EvaluationLoop(self.mock_generator, self.mock_analyzer)
        result = loop.run("3 apples", 3, mode="direct")
        
        self.assertTrue(result['match'])
        self.assertEqual(result['detected_count'], 3)
        self.mock_generator.generate.assert_called_once()
        self.mock_analyzer.analyze.assert_called_once()

    def test_loop_mode_retry(self):
        # First analysis returns 2 (wrong), second returns 3 (correct)
        self.mock_analyzer.analyze.side_effect = [2, 3]
        
        loop = EvaluationLoop(self.mock_generator, self.mock_analyzer, self.mock_editor)
        result = loop.run("3 apples", 3, mode="loop")
        
        self.assertTrue(result['match'])
        self.assertEqual(result['detected_count'], 3)
        self.mock_editor.edit.assert_called_once()
        self.assertEqual(len(result['steps']), 2) # Generate, Edit

if __name__ == '__main__':
    unittest.main()
