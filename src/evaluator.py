import logging
from typing import Optional
from .models import ImageGenerator, ImageEditor, ImageAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationLoop:
    def __init__(
        self, 
        generator: ImageGenerator, 
        analyzer: ImageAnalyzer, 
        editor: Optional[ImageEditor] = None,
        max_retries: int = 2
    ):
        self.generator = generator
        self.analyzer = analyzer
        self.editor = editor
        self.max_retries = max_retries

    def run(self, prompt: str, target_count: int, mode: str = "direct"):
        logger.info(f"Starting evaluation with mode={mode}, target_count={target_count}")
        
        # Step 1: Generate
        logger.info("Generating initial image...")
        result = self.generator.generate(prompt)
        logger.info(f"Image generated at {result.image_path}")
        
        # Step 2: Analyze
        count = self.analyzer.analyze(result.image_path, f"Count the number of {prompt.split()[-1]} in this image. Return only the number.")
        logger.info(f"Analysis result: {count}")
        
        if mode == "direct":
            return {
                "image_path": result.image_path,
                "target_count": target_count,
                "detected_count": count,
                "match": count == target_count,
                "steps": [{"action": "generate", "count": count}]
            }
            
        # Step 3: Loop (if mode is loop or two-pass)
        steps = [{"action": "generate", "count": count}]
        current_image_path = result.image_path
        
        retries = 0
        while count != target_count and retries < self.max_retries:
            if not self.editor:
                logger.warning("Editor not provided, cannot fix image.")
                break
                
            logger.info(f"Count mismatch ({count} != {target_count}). Attempting edit {retries + 1}/{self.max_retries}...")
            
            # Construct edit prompt
            edit_prompt = f"Make sure there are exactly {target_count} {prompt.split()[-1]} in the image."
            
            edit_result = self.editor.edit(current_image_path, edit_prompt)
            current_image_path = edit_result.image_path
            
            count = self.analyzer.analyze(current_image_path, f"Count the number of {prompt.split()[-1]} in this image. Return only the number.")
            logger.info(f"Analysis after edit: {count}")
            
            steps.append({"action": "edit", "count": count})
            retries += 1
            
        return {
            "image_path": current_image_path,
            "target_count": target_count,
            "detected_count": count,
            "match": count == target_count,
            "steps": steps
        }
