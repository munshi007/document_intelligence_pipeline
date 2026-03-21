"""
Reading Order VLM Planner
Uses a Vision Language Model to generate structural priors for a document page,
guiding the reading order resolution strategy.
"""

import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2

from common.vlm_client import VLMClient
from common.vlm_types import ReadingOrderPrior

logger = logging.getLogger(__name__)

class ReadingOrderPlannerVLM:
    """Uses a VLM to classify page layout and suggest a reading order strategy."""
    
    def __init__(self, vlm_client: Optional[VLMClient] = None):
        self.vlm_client = vlm_client
        
    def _prepare_image(self, page_image: np.ndarray) -> Image.Image:
        """Convert numpy image to PIL and resize for VLM."""
        if len(page_image.shape) == 3 and page_image.shape[2] == 3:
            pil_img = Image.fromarray(page_image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB))
            
        max_size = (1024, 1024)
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return pil_img

    def generate_priors(
        self, 
        page_image: np.ndarray,
        custom_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ReadingOrderPrior:
        """
        Analyze the page image and return layout priors.
        """
        if not self.vlm_client:
            logger.warning("VLM Client not provided. Returning default simple_linear prior.")
            return ReadingOrderPrior(
                reasoning="VLM client disabled, defaulting to simple heuristics.",
                layout_type="simple_linear",
                suggested_strategy="xy_cut"
            )
            
        prompt = custom_prompt or (
            "Analyze the layout of this document page to determine the correct reading order flow. "
            "Identify if the document is a 'simple_linear' page (like a letter or book), "
            "a 'multi_column' page (like an academic paper with 2 or 3 distinct columns), "
            "or a 'complex_unstructured' page (like a magazine, brochure, or complex form with sidebars or floating images). "
            "Based on your classification, suggest the 'suggested_strategy': "
            "Use 'xy_cut' for simple_linear. "
            "Use 'xy_cut_column_first' for multi_column to force a vertical split. "
            "Use 'deep_model' for complex_unstructured layouts."
        )
        
        logger.info("Sending page to VLM for reading order planning...")
        pil_img = self._prepare_image(page_image)
        
        result = self.vlm_client.generate_structured(
            image=pil_img,
            prompt=prompt,
            response_model=ReadingOrderPrior,
            metadata=metadata
        )
        
        if result is None:
            logger.warning("VLM Planner returned None. Defaulting to simple_linear.")
            return ReadingOrderPrior(
                reasoning="VLM classification failed, defaulting to simple heuristics.",
                layout_type="simple_linear",
                suggested_strategy="xy_cut"
            )
        
        logger.info(f"VLM Planner prioritized layout as '{result.layout_type}', strategy '{result.suggested_strategy}'")
        return result
