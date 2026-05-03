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
        image: Optional[Image.Image], 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False
    ) -> Optional[T]:
        content = []
        
        if image:
            base64_image, media_type = self._prepare_image(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image,
                },
            })
            
        schema = response_model.model_json_schema()
        system_prompt = (
            "You are a professional document analysis agent. "
            "Extract information into a valid JSON object matching the provided schema. "
            "Respond ONLY with the JSON block. Do not include markdown formatting."
        )
        
        user_prompt = f"{prompt}\n\nJSON SCHEMA:\n{json.dumps(schema, indent=2)}"
        content.append({"type": "text", "text": user_prompt})

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                timeout=self.timeout
            )
            
            res_text = message.content[0].text
            # Basic cleanup of markdown blocks if present
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif "```" in res_text:
                res_text = res_text.split("```")[1].strip()
            
            return response_model.model_validate_json(res_text)
        except Exception as e:
            logger.error(f"Anthropic Provider Error: {e}")
            return None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        is_complex: bool = False
    ) -> str:
        content = []
        
        if image:
            base64_image, media_type = self._prepare_image(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image
                }
            })
            
        content.append({"type": "text", "text": prompt})
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic Provider Error: {e}")
            return f"Error: {e}"
