import logging
import json
import re
from typing import Type, TypeVar, Optional, Dict, Any, List
from PIL import Image
import pydantic
import torch
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class LocalUnslothProvider(BaseVLMProvider):
    _model = None
    _tokenizer = None
    _current_path = None

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_path = model_name
        self._load_model()

    def _load_model(self):
        """Singleton-style model loader for Unsloth models."""
        if LocalUnslothProvider._model is None or LocalUnslothProvider._current_path != self.model_path:
            try:
                from unsloth import FastVisionModel
                logger.info(f"Loading local VLM from {self.model_path} with OOM fallback...")
                
                LocalUnslothProvider._model, LocalUnslothProvider._tokenizer = FastVisionModel.from_pretrained(
                    model_name=self.model_path,
                    load_in_4bit=True,
                    device_map="cuda:0",
                    trust_remote_code=True
                )
                FastVisionModel.for_inference(LocalUnslothProvider._model)
                LocalUnslothProvider._current_path = self.model_path
            except Exception as e:
                logger.error(f"Failed to load local Unsloth model: {e}")
                raise

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False
    ) -> Optional[T]:
        if not LocalUnslothProvider._model:
            return None

        schema = response_model.model_json_schema()
        
        # Consistent prompt structure
        full_prompt = f"""{prompt}

Respond in valid JSON format matching this schema:

SCHEMA:
{json.dumps(schema, indent=2)}

Format your response as:
Chain of Thought: <reasoning>
```json
<json_data>
```"""

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt}
            ]}
        ]

        try:
            input_text = LocalUnslothProvider._tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            inputs = LocalUnslothProvider._tokenizer(
                image, 
                input_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to("cuda")

            from transformers import TextStreamer
            outputs = LocalUnslothProvider._model.generate(
                **inputs,
                max_new_tokens=4096,
                use_cache=True,
                temperature=0.1,
                do_sample=False
            )
            
            decoded = LocalUnslothProvider._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', decoded, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return response_model.model_validate_json(json_str)
            
            return None
        except Exception as e:
            logger.error(f"Local Unsloth Provider Error: {e}")
            return None

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False
    ) -> str:
        if self.model is None or self.processor is None:
            return "Error: Local model not initialized"

        try:
            # Re-use private structured logic but without schema enforcement
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
            
            output = self.model.generate(
                **inputs, 
                max_new_tokens=4096,
                use_cache=True,
                do_sample=False
            )
            
            # Use processor's decode
            result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return result.strip()
            
        except Exception as e:
            logger.error(f"Local Unsloth Provider Error: {e}")
            return f"Error: {e}"
