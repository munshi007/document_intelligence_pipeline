import logging
import base64
import json
import re
import requests
import time
import os
from typing import Type, TypeVar, Optional, Dict, Any, Literal
from pydantic import BaseModel, ValidationError
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from config import VLM_CONFIG

# Load environment variables for API keys
load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class VLMClient:
    """
    Enterprise-Grade Client for interacting with VLM APIs (Ollama, OpenAI).
    Guarantees output matches the provided Pydantic schema or returns None.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # SOTA: Provider detection
        self.model = self.config.get('model', 'qwen2.5-vl:7b')
        self.provider = self.config.get('provider')
        self.observer = None # SOTA: Distillation Agent hook
        
        # SOTA: Auto-detect providers for research-worthy models
        if not self.provider:
            if self.model.startswith(('gpt-', 'o1-', 'o3-')):
                self.provider = 'openai'
            elif 'internvl' in self.model.lower():
                self.provider = 'internvl'
            elif 'got-ocr' in self.model.lower():
                self.provider = 'got-ocr'
            else:
                self.provider = 'ollama'
        
        # Base URLs
        if self.provider == 'ollama':
            self.base_url = self.config.get('base_url', 'http://localhost:11434/api')
        else:
            self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found in environment. OpenAI requests will fail.")

        # SOTA: Global configuration from pipeline_config.py
        self.timeout = self.config.get('timeout', VLM_CONFIG['timeout_seconds']) 
        self.max_image_dim = self.config.get('max_image_dim', VLM_CONFIG['max_image_res'])
        self.complex_res = self.config.get('complex_res', VLM_CONFIG['complex_image_res'])

    def _optimize_for_detail(self, image: Image.Image, is_complex: bool = False) -> str:
        """
        SOTA: Crop-to-Zoom Optimization.
        If is_complex=True (e.g. for small-text tables), we bypass heavy downsampling
        to preserve sub-pixel details required for research-grade OCR.
        """
        try:
            target_dim = self.complex_res if is_complex else self.max_image_dim
            
            # Preserve aspect ratio while hitting target
            if max(image.width, image.height) > target_dim:
                image.thumbnail((target_dim, target_dim), Image.Resampling.LANCZOS)
                
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=90) # Higher quality for complex regions
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"VLM Image optimization failed: {e}")
            return None

    def _extract_json_from_markdown(self, text: str) -> Optional[Dict]:
        """
        Extracts JSON from VLM markdown outputs to bypass tokenizer schema constraints.
        """
        try:
            # 1. Look for explicit ```json blocks
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 2. Fallback: find outer braces
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return json.loads(text[start_idx:end_idx+1])
                
            return None
        except (json.JSONDecodeError, ValueError):
            return None

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False,
        metadata: Optional[Dict] = None
    ) -> Optional[T]:
        """
        Agentic execution of VLM request with fallback guarantees.
        is_complex triggers 'Crop-to-Zoom' for higher resolution extraction.
        """
        # 1. Optimize Image (SOTA Resolution Scaling)
        base64_image = self._optimize_for_detail(image, is_complex)
        if not base64_image:
            logger.warning("VLM aborted: Image optimization failed.")
            return None
            
        # 2. Extract Schema for Prompting
        schema_dict = response_model.model_json_schema()
        
        # 3. Create SOTA Structured Markdown Prompt
        structured_prompt = f"""{prompt}

Review the document structurally. Then, you MUST output a valid JSON block containing your final answer, exactly matching this JSON Schema:

SCHEMA:
{json.dumps(schema_dict, indent=2)}

Respond EXACTLY in this markdown format:
Chain of Thought: <your reasoning here>
```json
<your valid json matching the schema here>
```"""

        result = None
        if self.provider == 'ollama':
            result = self._generate_ollama(structured_prompt, base64_image, response_model)
        elif self.provider == 'openai':
            result = self._generate_openai(structured_prompt, base64_image, response_model)
        elif self.provider == 'internvl':
            result = self._generate_internvl(structured_prompt, base64_image, response_model)
        elif self.provider == 'got-ocr':
            result = self._generate_got_ocr(structured_prompt, base64_image, response_model)
        else:
            logger.error(f"Unknown VLM provider: {self.provider}")
            return None

        # SOTA: Trigger Distillation Hook
        if result and self.observer:
            try:
                self.observer.capture(image, prompt, result, metadata=metadata)
            except Exception as e:
                logger.warning(f"Distillation capture failed: {e}")

        return result

    def _generate_ollama(self, prompt: str, base64_image: str, response_model: Type[T]) -> Optional[T]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Document AI agent. You act as an agentic router. You execute structural extraction perfectly and always output valid markdown JSON blocks."
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        
        url = f"{self.base_url.rstrip('/')}/chat"
        return self._execute_request(url, payload, response_model)

    def _generate_openai(self, prompt: str, base64_image: str, response_model: Type[T]) -> Optional[T]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Document AI agent. You act as an agentic router. You execute structural extraction perfectly and always output valid markdown JSON blocks."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0
        }
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return self._execute_request(url, payload, response_model, headers)

    def _generate_internvl(self, prompt: str, base64_image: str, response_model: Type[T]) -> Optional[T]:
        """
        SOTA: InternVL2 Adapter.
        Uses specialized prompt formatting for InternVL's multi-modal architecture.
        """
        # InternVL via standard OpenAI-compatible API (like VLLM or LMDeploy)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<image>\n{prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        return self._execute_request(url, payload, response_model)

    def _generate_got_ocr(self, prompt: str, base64_image: str, response_model: Type[T]) -> Optional[T]:
        """
        SOTA: GOT-OCR2.0 Adapter.
        Optimized for high-resolution table and formula extraction.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"OCR with format: {prompt}",
                    "images": [base64_image]
                }
            ]
        }
        url = f"{self.base_url.rstrip('/')}/chat" # Standardized endpoint
        return self._execute_request(url, payload, response_model)

    def _execute_request(self, url: str, payload: Dict, response_model: Type[T], headers: Optional[Dict] = None) -> Optional[T]:
        logger.info(f"Firing VLM request to {self.model} via {self.provider} (Timeout SLA: {self.timeout}s)...")
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                resp_json = response.json()
                if self.provider == 'ollama':
                    result_text = resp_json.get('message', {}).get('content', '')
                else:
                    result_text = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                parsed_dict = self._extract_json_from_markdown(result_text)
                if parsed_dict:
                    try:
                        validated_model = response_model(**parsed_dict)
                        logger.info(f"✅ VLM Success ({self.provider}) in {duration:.2f}s")
                        return validated_model
                    except ValidationError as ve:
                        logger.error(f"❌ VLM Schema Validation Failed: {ve}")
                        return None
                else:
                    logger.error(f"❌ VLM JSON Parsing Failed in {duration:.2f}s. Output:\n{result_text[:200]}")
                    return None
            else:
                logger.error(f"❌ VLM HTTP Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ VLM TIMEOUT ({self.timeout}s exceeded). Graceful degradation triggered.")
            return None
        except Exception as e:
            logger.error(f"❌ VLM Fatal Error: {e}")
            return None
