import os
import base64
import requests
from .models import ImageGenerator, ImageResult
from google import genai
from google.genai import types
from openai import OpenAI
import fal_client
from .utils import get_output_path

class GeminiGenerator(ImageGenerator):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1beta'})

    def generate(self, prompt: str) -> ImageResult:
        try:
            response = self.client.models.generate_content(
                model='gemini-3-pro-image-preview',
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )
            
            # Save image
            image_path = get_output_path("generated_gemini")
            
            # Search for inline data
            image_saved = False
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        try:
                            # Attempt to save using SDK helper if available
                            part.as_image().save(image_path)
                            image_saved = True
                            break
                        except:
                            # Fallback to manual decode
                            import base64
                            with open(image_path, "wb") as f:
                                f.write(base64.b64decode(part.inline_data.data))
                            image_saved = True
                            break
            
            if not image_saved:
                 raise ValueError("No image part found in Gemini response")
            
            return ImageResult(image_path=image_path, metadata={"model": "gemini-3-pro-image-preview"})
        except Exception as e:
            # Fallback logic or detailed error
            raise RuntimeError(f"Gemini generation failed: {e}")

class OpenAIGenerator(ImageGenerator):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> ImageResult:
        try:
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                size="1024x1024",
                # response_format="b64_json" # Removed as it might not be supported for this model yet or defaults differently
            )
            
            # Handle response if it's a URL (common default) or b64
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                image_data = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], 'url') and response.data[0].url:
                image_data = requests.get(response.data[0].url).content
            else:
                raise ValueError("No image data found in response")

            image_path = get_output_path("generated_openai")
            with open(image_path, "wb") as f:
                f.write(image_data)
                
            return ImageResult(image_path=image_path, metadata={"model": "gpt-image-1"})
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

class FalGenerator(ImageGenerator):
    def __init__(self, model_id: str = "fal-ai/recraft-v3"):
        self.model_id = model_id

    def generate(self, prompt: str) -> ImageResult:
        try:
            handler = fal_client.submit(
                self.model_id,
                arguments={"prompt": prompt}
            )
            result = handler.get()
            
            image_url = result['images'][0]['url']
            image_data = requests.get(image_url).content
            image_path = get_output_path(f"generated_fal_{self.model_id.replace('/', '_')}")
            with open(image_path, "wb") as f:
                f.write(image_data)
                
            return ImageResult(image_path=image_path, metadata={"model": self.model_id})
        except Exception as e:
             raise RuntimeError(f"Fal generation failed: {e}")
