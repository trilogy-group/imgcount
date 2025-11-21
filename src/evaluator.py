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

    def run(self, prompt: str, target_count: int, mode: str = "direct", object_name: Optional[str] = None):
        logger.info(f"Starting evaluation with mode={mode}, target_count={target_count}, object_name={object_name}")
        
        target_object = object_name if object_name else prompt.split()[-1]

        # Step 1: Generate
        logger.info("Generating initial image...")
        result = self.generator.generate(prompt)
        logger.info(f"Image generated at {result.image_path}")
        
        # Step 2: Analyze
        count_prompt = f"Count the number of {target_object} in this image. Return a JSON object with a single key 'count' and the integer value."
        count = self.analyzer.analyze(result.image_path, count_prompt)
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
            edit_prompt = f"Make sure there are exactly {target_count} {target_object} in the image."
            
            try:
                edit_result = self.editor.edit(current_image_path, edit_prompt)
                current_image_path = edit_result.image_path
                
                count = self.analyzer.analyze(current_image_path, count_prompt)
                logger.info(f"Analysis after edit: {count}")
                
                steps.append({"action": "edit", "count": count})
                retries += 1
            except Exception as e:
                logger.error(f"Edit failed: {e}")
                break
            
        return {
            "image_path": current_image_path,
            "target_count": target_count,
            "detected_count": count,
            "match": count == target_count,
            "steps": steps
        }
