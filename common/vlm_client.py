import logging
import os
import io
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic
from dotenv import load_dotenv

from .vlm_providers.openai_provider import OpenAIProvider
from .vlm_providers.anthropic_provider import AnthropicProvider
from .vlm_providers.ollama_provider import OllamaProvider
from .vlm_providers.local_unsloth_provider import LocalUnslothProvider
from config import VLM_CONFIG

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class VLMClient:
    """
    Modular Client for interacting with various VLM providers.
    Uses specialized adapters for OpenAI, Anthropic, Ollama, and Local models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get('model', 'qwen2.5-vl:7b')
        self.provider_name = self.config.get('provider')
        self.observer = None
        
        # 1. Auto-detect provider if not explicitly given
        if not self.provider_name:
            if self.model.startswith(('gpt-', 'o1-', 'o3-')):
                self.provider_name = 'openai'
            elif self.model.startswith('claude-'):
                self.provider_name = 'anthropic'
            elif os.path.isdir(self.model) or '/' in self.model: 
                self.provider_name = 'local'
            else:
                self.provider_name = 'ollama'
        
        # 2. Load API keys from environment
        self.api_key = self._get_api_key()
        
        # 3. Initialize the specific provider adapter
        try:
            self.provider = self._init_provider()
        except Exception as e:
            logger.error(f"Failed to initialize VLM provider '{self.provider_name}': {e}")
            self.provider = None
        
        # Global configuration
        self.max_image_dim = VLM_CONFIG['max_image_res']
        self.complex_res = self.config.get('complex_res', VLM_CONFIG['complex_image_res'])

    def _get_api_key(self) -> Optional[str]:
        if self.provider_name == 'openai':
            return os.getenv("OPENAI_API_KEY")
        elif self.provider_name == 'anthropic':
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def _init_provider(self):
        kwargs = {"timeout": self.config.get('timeout', VLM_CONFIG['timeout_seconds'])}
        
        if self.provider_name == 'openai':
            return OpenAIProvider(self.model, self.api_key, **kwargs)
        elif self.provider_name == 'anthropic':
            return AnthropicProvider(self.model, self.api_key, **kwargs)
        elif self.provider_name == 'ollama':
            return OllamaProvider(self.model, **kwargs)
        elif self.provider_name == 'local':
            return LocalUnslothProvider(self.model, **kwargs)
        else:
            raise ValueError(f"Unsupported VLM provider: {self.provider_name}")

    def _optimize_for_detail(self, image: Image.Image, is_complex: bool = False) -> Image.Image:
        """
        Resize image while preserving resolution for complex regions.
        """
        try:
            target_dim = self.complex_res if is_complex else self.max_image_dim
            w, h = image.size
            if max(w, h) > target_dim:
                scale = target_dim / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                return image.resize(new_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return image

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False,
        metadata: Optional[Dict] = None
    ) -> Optional[T]:
        """
        Standardized entry point for structured extraction.
        """
        if not self.provider:
            logger.error("VLM provider not initialized.")
            return None

        # 1. Optimize Image
        optimized_image = self._optimize_for_detail(image, is_complex)
        
        # 2. Delegate to specific provider
        result = self.provider.generate_structured(
            image=optimized_image,
            prompt=prompt,
            response_model=response_model,
            is_complex=is_complex
        )
        
        # 3. Capture for distillation if observer is present
        if result and self.observer:
            try:
                self.observer.capture(image, prompt, result, metadata=metadata)
            except Exception as e:
                logger.warning(f"Distillation capture failed: {e}")
                
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Standardized entry point for raw text/markdown generation.
        """
        if not self.provider:
            logger.error("VLM provider not initialized.")
            return "Error: Provider not initialized"

        # 1. Optimize Image
        optimized_image = self._optimize_for_detail(image, is_complex)

        # 2. Delegate to specific provider
        result = self.provider.generate(
            image=optimized_image,
            prompt=prompt,
            is_complex=is_complex
        )

        return result
