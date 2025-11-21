import os
import base64
import requests
from .models import ImageAnalyzer
from google import genai
from google.genai import types
from openai import OpenAI

class QwenAnalyzer(ImageAnalyzer):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.model = "qwen/qwen3-vl-235b-a22b-instruct"

    def analyze(self, image_path: str, prompt: str) -> int:
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            
            content = response.choices[0].message.content
            # Simple parsing for now, assuming the model returns a number or we extract it
            # The prompt should ask for a single number.
            
            # Try to parse JSON if present
            import json
            import re
            
            # Check for JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if 'count' in data:
                        return int(data['count'])
                except:
                    pass

            # Fallback to simple number extraction
            numbers = re.findall(r'\d+', content)
            if numbers:
                return int(numbers[0])
            return -1 # Error code
        except Exception:
            return -1

class GeminiAnalyzer(ImageAnalyzer):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1beta'})

    def analyze(self, image_path: str, prompt: str) -> int:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                
            response = self.client.models.generate_content(
                model='gemini-3-pro', # Updated to Gemini 3 Pro
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
            )
            
            content = response.text
            import json
            import re

            # Check for JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if 'count' in data:
                        return int(data['count'])
                except:
                    pass
            
            numbers = re.findall(r'\d+', content)
            if numbers:
                return int(numbers[0])
            return -1
        except Exception:
            return -1
