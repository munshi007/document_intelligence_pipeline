import logging
import json
import base64
import io
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic
from openai import OpenAI
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class OpenAIProvider(BaseVLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.timeout = kwargs.get('timeout', 120)

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
        
        system_prompt = (
            "You are a professional document analysis agent. "
            "Extract information into a valid JSON object matching the provided schema."
        )
        
        user_prompt = f"{prompt}\n\nJSON SCHEMA:\n{json.dumps(schema, indent=2)}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content
            return response_model.model_validate_json(content)
        except Exception as e:
            logger.error(f"OpenAI Provider Error: {e}")
            return None

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False
    ) -> str:
        base64_image = self._prepare_image(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Provider Error: {e}")
            return f"Error: {e}"
