import os
import base64
import requests
from .models import ImageEditor, ImageResult
from google import genai
from google.genai import types
from openai import OpenAI
import fal_client
from .utils import get_output_path

class GeminiEditor(ImageEditor):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1beta'})

    def edit(self, image_path: str, prompt: str) -> ImageResult:
        # Placeholder for Gemini editing if available. 
        # Currently Gemini API primarily supports generation. 
        # We might need to use a specific endpoint or wait for support.
        # For now, we will raise NotImplementedError or try to use a generation with image input if supported.
        # Assuming 'gemini-2.5-flash-image' might support image-to-image or we use a different approach.
        
        # Reading the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        # TODO: Check actual Gemini Edit API support for gemini-2.5-flash-image. 
        # For now, returning a dummy result to allow flow testing if needed, 
        # or raising error.
        raise NotImplementedError("Gemini Editing not yet fully implemented/verified.")

class OpenAIEditor(ImageEditor):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def edit(self, image_path: str, prompt: str) -> ImageResult:
        try:
            # OpenAI Edit uses GPT Image 1
            response = self.client.images.edit(
                model="gpt-image-1",
                image=open(image_path, "rb"),
                prompt=prompt,
                n=1,
                size="1024x1024",
                # response_format="b64_json"
            )
            
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                image_data = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], 'url') and response.data[0].url:
                image_data = requests.get(response.data[0].url).content
            else:
                 raise ValueError("No image data found in response")
                 
            new_image_path = get_output_path(f"edited_openai_{os.path.basename(image_path).split('.')[0]}")
            with open(new_image_path, "wb") as f:
                f.write(image_data)
                
            return ImageResult(image_path=new_image_path, metadata={"model": "gpt-image-1"})
        except Exception as e:
            raise RuntimeError(f"OpenAI edit failed: {e}")

class FalEditor(ImageEditor):
    def __init__(self, model_id: str = "fal-ai/recraft-v3"):
        self.model_id = model_id

    def edit(self, image_path: str, prompt: str) -> ImageResult:
        try:
            # Upload image to Fal storage first (or use data URI if supported, but Fal usually likes URLs)
            url = fal_client.upload_file(image_path)
            
            # Recraft V3 Edit arguments might differ, checking generic structure
            # Usually takes 'image_url' and 'prompt'
            arguments = {
                "prompt": prompt,
                "image_url": url
            }
            
            # Adjust arguments based on specific model requirements if needed
            if "recraft" in self.model_id:
                 # Recraft might need specific style or other params
                 pass

            handler = fal_client.submit(
                self.model_id,
                arguments=arguments
            )
            result = handler.get()
            
            image_url = result['images'][0]['url']
            image_data = requests.get(image_url).content
            new_image_path = get_output_path(f"edited_fal_{os.path.basename(image_path).split('.')[0]}")
            with open(new_image_path, "wb") as f:
                f.write(image_data)
                
            return ImageResult(image_path=new_image_path, metadata={"model": self.model_id})
        except Exception as e:
            raise RuntimeError(f"Fal edit failed: {e}")
