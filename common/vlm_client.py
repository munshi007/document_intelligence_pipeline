import logging
import os
import io
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic
from dotenv import load_dotenv

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
        # Finalized Production Model Defaults
        self.model = self.config.get('model', 'models/custom')
        self.provider_name = 'local'
        self.observer = None
        
        # Initialize the specific provider adapter
        try:
            self.provider = LocalUnslothProvider(self.model)
        except Exception as e:
            logger.error(f"Failed to initialize Local Specialist VLM: {e}")
            self.provider = None
        
        # Global configuration
        self.max_image_dim = VLM_CONFIG['max_image_res']
        self.complex_res = self.config.get('complex_res', VLM_CONFIG['complex_image_res'])

    def _get_api_key(self) -> Optional[str]:
        return None

    def _init_provider(self):
        return LocalUnslothProvider(self.model)

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
