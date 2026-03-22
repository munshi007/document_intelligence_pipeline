from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Dict, Any
from PIL import Image
import pydantic

T = TypeVar("T", bound=pydantic.BaseModel)

class BaseVLMProvider(ABC):
    """
    Abstract interface for all VLM providers.
    Ensures a standardized 'generate_structured' flow.
    """
    
    @abstractmethod
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        pass

    @abstractmethod
    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False
    ) -> Optional[T]:
        """Execute request and return structured data matching response_model."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False
    ) -> str:
        """Execute request and return raw text/markdown response."""
        pass
