"""
VLM Stylesheet Planner
Uses a Vision Language Model to hypothesize a DocumentStyleSheet based on visual page structure,
and grounds the VLM's guesses against deterministic pdfplumber font statistics.
"""

from typing import Optional
import logging

from PIL import Image

from common.vlm_client import VLMClient
from common.vlm_types import DocumentStyleSheet, FontSignature
from processors.font_analyzer import FontAnalyzer

logger = logging.getLogger(__name__)

class StylesheetPlanner:
    def __init__(self, vlm_client: Optional[VLMClient] = None):
        self.vlm_client = vlm_client or VLMClient()
        
    def generate_hypothesized_stylesheet(self, page_image: Image.Image) -> Optional[DocumentStyleSheet]:
        """
        Asks the VLM to visually hypothesize the semantic stylesheet for the document.
        """
        prompt = (
            "Analyze the layout and typography of this document page. "
            "Identify the visual hierarchy: title, headers (H1, H2, H3), body text, and captions. "
            "For each style, estimate its relative font size (e.g., body=12, H1=18), whether it is bold or italic, "
            "and note any distinct colors. Do not hallucinate content; focus purely on the visual structure."
        )
        
        stylesheet = self.vlm_client.generate_structured(
            image=page_image,
            prompt=prompt,
            response_model=DocumentStyleSheet
        )
        return stylesheet

    def ground_stylesheet(self, vlm_stylesheet: Optional[DocumentStyleSheet], font_analyzer: FontAnalyzer, page_limit: int = 3) -> DocumentStyleSheet:
        """
        Grounds the VLM's hypothesized stylesheet against the physical font statistics
        extracted by pdfplumber (the deterministic fallback stats).
        """
        deterministic_fallback = font_analyzer.infer_stylesheet_from_stats(page_limit=page_limit)
        
        if not vlm_stylesheet:
            logger.warning("VLM Stylesheet generation offline. Using deterministic fallback.")
            return deterministic_fallback
            
        logger.info("Grounding VLM stylesheet against deterministic font physics.")
        
        grounded = DocumentStyleSheet(
            reasoning=f"VLM reasoning: {vlm_stylesheet.reasoning} | Grounded mathematically.",
            body=deterministic_fallback.body
        )
        
        # Snap headers to physical truth
        if deterministic_fallback.h1:
            grounded.h1 = deterministic_fallback.h1
        elif vlm_stylesheet.h1:
            grounded.h1 = vlm_stylesheet.h1
            
        if deterministic_fallback.h2:
            grounded.h2 = deterministic_fallback.h2
        elif vlm_stylesheet.h2:
            grounded.h2 = vlm_stylesheet.h2
            
        if deterministic_fallback.h3:
            grounded.h3 = deterministic_fallback.h3
        elif vlm_stylesheet.h3:
            grounded.h3 = vlm_stylesheet.h3
            
        grounded.title = vlm_stylesheet.title
        grounded.caption = vlm_stylesheet.caption
        
        return grounded
