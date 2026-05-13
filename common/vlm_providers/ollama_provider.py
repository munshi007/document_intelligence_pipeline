import logging
import json
import base64
import io
import requests
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class OllamaProvider(BaseVLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.base_url = kwargs.get('base_url', "http://localhost:11434/api/generate")

    def _prepare_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False
    ) -> Optional[T]:
        base64_image = self._prepare_image(image)
        schema = response_model.model_json_schema()
        
        full_prompt = f"""{prompt}
        
        Analyze the image and return ONLY a JSON object with this exact keys:
        {json.dumps(list(schema['properties'].keys()))}
        
        Respond with raw JSON only."""
    

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": [base64_image],
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "")
            return response_model.model_validate_json(content)
        except pydantic.ValidationError as e:
            logger.warning(f"Ollama Provider: VLM generated invalid JSON structure. Using deterministic fallback.")
            return None
        except Exception as e:
            logger.error(f"Ollama Provider API Error: {e}")
            return None

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False
    ) -> str:
        base64_image = self._prepare_image(image)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama Provider Error: {e}")
            return f"Error: {e}"
