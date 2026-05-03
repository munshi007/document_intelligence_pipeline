"""
Reading Order VLM Referee
Uses a VLM to QA test the final markdown text produced by the pipeline,
ensuring sentences are not split across columns and reading order is coherent.
"""

import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2

from common.vlm_client import VLMClient
from common.vlm_types import ReadingOrderVerification

logger = logging.getLogger(__name__)

class ReadingOrderRefereeVLM:
    """Uses a VLM to QA test stitched reading order."""
    
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

    def verify_order(
        self, 
        page_image: np.ndarray,
        extracted_markdown: str,
        custom_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ReadingOrderVerification:
        """
        Analyze the extracted markdown against the page image to verify reading order.
        
        Args:
            page_image: The numpy array of the page image.
            extracted_markdown: The stitched text to verify.
            
        Returns:
            ReadingOrderVerification: QA results with suggested actions.
        """
        if not self.vlm_client:
            logger.warning("VLM Client not provided. Auto-accepting reading order.")
            return ReadingOrderVerification(
                reasoning="VLM client disabled.",
                is_continuous=True,
                score=10,
                suggested_action="accept"
            )
            
        # Truncate markdown to avoid token bloat for simple QA
        truncated_md = extracted_markdown[:2000] + ("..." if len(extracted_markdown) > 2000 else "")
            
        prompt = custom_prompt or (
            "You are a Quality Assurance agent verifying PDF reading order extraction. "
            "Examine this image and the extracted text below. "
            "Your job is to determine if the extracted text flows logically according to natural human reading order. "
            "Look for 'semantic breaks' — e.g. a sentence in column 1 that is suddenly interrupted by a paragraph from column 2. "
            f"\\n\\n--- EXTRACTED TEXT ---\\n{truncated_md}\\n--- END TEXT ---\\n\\n"
            "If the text is continuous and correct, suggest 'accept'. "
            "If the columns are interleaved (mixed horizontally), suggest 'rerun_column_first'. "
            "If it is completely jumbled beyond repair, suggest 'escalate'."
        )
        
        logger.info("Sending stitched text to VLM Referee for reading order QA...")
        pil_img = self._prepare_image(page_image)
        
        result = self.vlm_client.generate_structured(
            image=pil_img,
            prompt=prompt,
            response_model=ReadingOrderVerification,
            metadata=metadata
        )
        
        if result is None:
            logger.info("VLM Referee: No vision feedback. Auto-accepting reading order.")
            return ReadingOrderVerification(
                reasoning="VLM evaluation failed, auto-accepting.",
                is_continuous=True,
                score=5,
                suggested_action="accept"
            )
        
        logger.info(f"Referee scored {result.score}/10, action: {result.suggested_action}")
        return result
