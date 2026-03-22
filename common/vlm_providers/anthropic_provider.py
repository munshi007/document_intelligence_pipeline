import logging
import json
import base64
import io
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic
import anthropic
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class AnthropicProvider(BaseVLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.timeout = kwargs.get('timeout', 120)

    def _prepare_image(self, image: Image.Image) -> tuple[str, str]:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), "image/png"

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False
    ) -> Optional[T]:
        base64_image, media_type = self._prepare_image(image)
        schema = response_model.model_json_schema()
        
        system_prompt = (
            "You are a professional document analysis agent. "
            "Extract information into a valid JSON object matching the provided schema. "
            "Respond ONLY with the JSON block."
        )
        
        user_prompt = f"{prompt}\n\nJSON SCHEMA:\n{json.dumps(schema, indent=2)}"

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": user_prompt}
                        ],
                    }
                ],
                timeout=self.timeout
            )
            
            # Claude doesn't have a rigid response_format json yet, so we parse
            content = message.content[0].text
            return response_model.model_validate_json(content)
        except Exception as e:
            logger.error(f"Anthropic Provider Error: {e}")
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
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic Provider Error: {e}")
            return f"Error: {e}"
