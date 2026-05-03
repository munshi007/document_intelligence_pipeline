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
            "Analyze the layout and typography of this document page to identify its DocumentStyleSheet.\n"
            "Identify the visual hierarchy: title, h1, h2, h3, body, and caption.\n\n"
            "### JSON STRUCTURE CONSTRAINTS:\n"
            "1. Each style field (like 'title' or 'body') MUST be an object { } with these fields:\n"
            "   - 'size': numeric float\n"
            "   - 'fontname': string name\n"
            "   - 'is_bold': boolean\n"
            "   - 'is_italic': boolean\n"
            "   - 'color': hex string or null\n"
            "2. DO NOT output a simple string for these fields.\n"
            "3. Focus on relative visual hierarchy (e.g. h1 is larger and bolder than body).\n"
            "4. Provide 'reasoning' as a separate top-level field."
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
            logger.info("VLM Stylesheet: No vision feedback (Offline). Using deterministic font analyzer.")
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
