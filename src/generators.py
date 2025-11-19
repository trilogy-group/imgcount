import os
import base64
import requests
from .models import ImageGenerator, ImageResult
from google import genai
from google.genai import types
from openai import OpenAI
import fal_client

class GeminiGenerator(ImageGenerator):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1alpha'})

    def generate(self, prompt: str) -> ImageResult:
        response = self.client.models.generate_images(
            model='gemini-2.5-flash-image',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        
        # Save image
        image_path = "generated_gemini.png"
        response.generated_images[0].image.save(image_path)
        
        return ImageResult(image_path=image_path, metadata={"model": "gemini-2.5-flash-image"})

class OpenAIGenerator(ImageGenerator):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> ImageResult:
        response = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json"
        )
        
        image_data = base64.b64decode(response.data[0].b64_json)
        image_path = "generated_openai.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        return ImageResult(image_path=image_path, metadata={"model": "gpt-image-1"})

class FalGenerator(ImageGenerator):
    def __init__(self, model_id: str = "fal-ai/recraft-v3"):
        self.model_id = model_id

    def generate(self, prompt: str) -> ImageResult:
        handler = fal_client.submit(
            self.model_id,
            arguments={"prompt": prompt}
        )
        result = handler.get()
        
        image_url = result['images'][0]['url']
        image_data = requests.get(image_url).content
        image_path = f"generated_fal_{self.model_id.replace('/', '_')}.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        return ImageResult(image_path=image_path, metadata={"model": self.model_id})
