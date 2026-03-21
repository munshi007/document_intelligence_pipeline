"""
VLM Table Planner - Use a Vision-Language Model to generate extraction priors.

This module calls a local VLM (via Ollama) to analyze a page thumbnail
and output priors that bias the deterministic router.

The VLM outputs PRIORS only, not extraction decisions.
"""

import logging
import json
import base64
from typing import Dict, Optional, Any
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TablePlannerVLM:
    """
    Generate table extraction priors using a Vision-Language Model.
    Uses centralized VLMClient for structured output.
    """
    
    DEFAULT_PROMPT = """Analyze this document page region and identify the table structure.
    Determine if this is a ruled table (visible grid), a key-value list (2-column), or a complex table.
    Assess the best strategy to extract it."""

    def __init__(
        self,
        vlm_client: Any = None,
    ):
        """
        Args:
            vlm_client: Instance of common.vlm_client.VLMClient
        """
        self.vlm_client = vlm_client
    
    def generate_priors(
        self,
        page_image,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Generate extraction priors from a page image.
        
        Args:
            page_image: PIL Image or numpy array
            custom_prompt: Optional custom prompt
        
        Returns:
            Dict with likelihood scores and boolean flags (compatible with router)
        """
        if not self.vlm_client:
            logger.warning("VLM client not provided to TablePlannerVLM")
            return self._default_priors()
            
        # Prepare image (PIL)
        image = self._to_pil(page_image)
        if image is None:
            return self._default_priors()
            
        prompt = custom_prompt or self.DEFAULT_PROMPT
        
        try:
            # Call VLM with structured output
            from common.vlm_types import TablePrior
            
            result = self.vlm_client.generate_structured(
                image=image,
                prompt=prompt,
                response_model=TablePrior,
                is_complex=True # SOTA: Tables require high-res grounding
            )
            
            if not result:
                return self._default_priors()
                
            # Convert Pydantic model to the dict format expected by the Router
            # Mapping logic:
            # - If suggests "ruled_vector", high ruled_likelihood
            # - If suggests "text_cluster", high kv_likelihood
            
            priors = {
                "kv_likelihood": 0.1,
                "ruled_likelihood": 0.1,
                "complex_likelihood": 0.1,
                "multi_column": False, # TODO: Add this to model if needed
                "reasoning": result.reasoning
            }
            
            if result.suggested_strategy == "ruled_vector":
                priors["ruled_likelihood"] = 0.9
            elif result.suggested_strategy == "text_cluster":
                priors["kv_likelihood"] = 0.9
            elif result.suggested_strategy == "hybrid" or result.table_type == "complex":
                priors["complex_likelihood"] = 0.9
                
            return priors
            
        except Exception as e:
            logger.warning(f"VLM Planner failed: {e}")
            return self._default_priors()

    def _to_pil(self, image) -> Optional[Image.Image]:
        """Convert numpy array to PIL Image."""
        if not PIL_AVAILABLE:
            return None
        try:
            if hasattr(image, "shape"):
                import numpy as np
                if isinstance(image, np.ndarray):
                    return Image.fromarray(image)
            return image
        except Exception:
            return None

    def _default_priors(self) -> Dict[str, float]:
        """Return default priors when VLM is unavailable."""
        return {
            "kv_likelihood": 0.33,
            "ruled_likelihood": 0.33,
            "complex_likelihood": 0.33,
            "multi_column": False,
        }


class TableRefereeVLM:
    """
    VLM-based referee for failed table extractions.
    Uses centralized VLMClient for structured output.
    """
    
    DEFAULT_PROMPT = """This table extraction failed quality checks.
    
    Stats: {stats}
    Preview (first 3 rows):
    {preview}
    
    Analyze the table image and the extraction quality.
    Decide if we should ACCEPT it (if errors are minor/irrelevant) or RERUN with a different strategy.
    """

    def __init__(
        self,
        vlm_client: Any = None,
    ):
        self.vlm_client = vlm_client
        # Planner helper not needed for image prep anymore as client handles encoding
    
    def suggest_action(
        self,
        table_image,
        qa_stats: Dict,
        preview_rows: list,
    ) -> Dict:
        """
        Get action suggestion from VLM.
        
        Args:
            table_image: Cropped table image (PIL or numpy)
            qa_stats: QA metrics dict
            preview_rows: First 3 rows of extracted table
        
        Returns:
            Action dict like {"action": "rerun", "strategy": "kv"}
        """
        if not self.vlm_client:
            return {"action": "escalate", "strategy": "tsr", "reason": "VLM unavailable"}
            
        # Prepare image
        image = self._to_pil(table_image)
        if image is None:
            return {"action": "escalate", "strategy": "tsr", "reason": "Image prep failed"}
        
        # Format prompt
        stats_str = json.dumps({k: round(v, 3) if isinstance(v, float) else v 
                               for k, v in qa_stats.items() if k != "unassigned_word_ids"})
        # Simple text preview
        preview_str = ""
        for i, row in enumerate(preview_rows):
            preview_str += f"Row {i}: {row}\n"
        
        prompt = self.DEFAULT_PROMPT.format(stats=stats_str, preview=preview_str)
        
        try:
            from common.vlm_types import TableVerification
            
            result = self.vlm_client.generate_structured(
                image=image,
                prompt=prompt,
                response_model=TableVerification,
                is_complex=True # SOTA: QA verification needs pixel-perfect detail
            )
            
            if not result:
                return {"action": "escalate", "strategy": "tsr", "reason": "No VLM response"}
            
            # Map structured output to action dict
            action_map = {
                "accept": "accept",
                "rerun_kv": "rerun",
                "rerun_ruled": "rerun",
                "rerun_complex": "rerun",
                "escalate": "escalate"
            }
            
            strategy_map = {
                "rerun_kv": "kv",
                "rerun_ruled": "ruled",
                "rerun_complex": "tsr",
            }
            
            return {
                "action": action_map.get(result.suggested_action, "escalate"),
                "strategy": strategy_map.get(result.suggested_action, "tsr"),
                "reason": result.reasoning
            }
            
        except Exception as e:
            logger.warning(f"Referee VLM call failed: {e}")
            return {"action": "escalate", "strategy": "tsr", "reason": str(e)}

    def _to_pil(self, image) -> Optional[Image.Image]:
        """Convert numpy array to PIL Image."""
        if not PIL_AVAILABLE:
            return None
        try:
            if hasattr(image, "shape"):
                import numpy as np
                if isinstance(image, np.ndarray):
                    return Image.fromarray(image)
            return image
        except Exception:
            return None
