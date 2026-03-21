"""
LayoutLMv3 Classifier - Smart text classification and relationship detection
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import LayoutLMv3
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from PIL import Image
    import torch
    LAYOUTLM_AVAILABLE = True
    logger.info("LayoutLMv3 dependencies available")
except ImportError:
    LAYOUTLM_AVAILABLE = False
    logger.warning("LayoutLMv3 not available - install transformers and torch")


class LayoutLMClassifier:
    """
    Use LayoutLMv3 for smart text classification and relationship detection.
    Falls back to rule-based methods if model not available.
    
    Based on the implementation from FINAL_TRY/Support-Code/ocr.py
    """
    
    def __init__(self, use_layoutlm: bool = True):
        """
        Initialize LayoutLM classifier.
        
        Args:
            use_layoutlm: Whether to use LayoutLMv3 (if available)
        """
        self.use_layoutlm = use_layoutlm and LAYOUTLM_AVAILABLE
        self.processor = None
        self.model = None
        
        if self.use_layoutlm:
            try:
                logger.info("Loading LayoutLMv3 model (microsoft/layoutlmv3-base)...")
                self.processor = LayoutLMv3Processor.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    apply_ocr=False  # We already have OCR results from PaddleOCR
                )
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    num_labels=4  # heading, label, paragraph, caption
                )
                self.model.eval()
                logger.info("LayoutLMv3 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LayoutLMv3: {e}")
                self.use_layoutlm = False
        else:
            logger.info("LayoutLMv3 not used - using rule-based fallback")
    
    def classify_text_region(
        self,
        text: str,
        bbox: List[float],
        page_image: Optional[np.ndarray] = None,
        font_info: Optional[Dict] = None
    ) -> str:
        """
        Classify text region as heading, label, paragraph, or caption.
        
        Args:
            text: Text content
            bbox: Bounding box [x0, y0, x1, y1]
            page_image: Page image (optional, for LayoutLMv3)
            font_info: Font information (for fallback)
            
        Returns:
            Classification: "heading", "label", "paragraph", "caption"
        """
        # Use LayoutLMv3 if available and image provided
        if self.use_layoutlm and page_image is not None:
            try:
                return self._classify_with_layoutlm(text, bbox, page_image)
            except Exception as e:
                logger.warning(f"LayoutLMv3 classification failed: {e}, using fallback")
        
        # Fallback to rule-based classification
        return self._classify_rule_based(text, font_info)
    
    def detect_relationship(
        self,
        text_region: Dict[str, Any],
        figure_regions: List[Dict[str, Any]],
        page_image: Optional[np.ndarray] = None
    ) -> str:
        """
        Detect relationship between text and figures.
        
        Args:
            text_region: Text region dict with 'text' and 'bbox'
            figure_regions: List of figure regions
            page_image: Page image (optional, for LayoutLMv3)
            
        Returns:
            Relationship: "inside_figure", "caption", "separate"
        """
        text_bbox = text_region.get('bbox')
        if not text_bbox or not figure_regions:
            return "separate"
        
        # Check spatial relationship first
        for figure in figure_regions:
            figure_bbox = figure.get('bbox')
            if not figure_bbox:
                continue
            
            iou = self._compute_iou(text_bbox, figure_bbox)
            
            # High overlap - likely inside figure
            if iou > 0.7:
                # Use LayoutLMv3 to confirm if available
                if self.use_layoutlm and page_image is not None:
                    try:
                        relationship = self._detect_relationship_with_layoutlm(
                            text_region, figure, page_image
                        )
                        return relationship
                    except Exception as e:
                        logger.warning(f"LayoutLMv3 relationship detection failed: {e}")
                
                # Fallback: assume inside figure
                return "inside_figure"
            
            # Medium overlap - might be caption
            elif 0.1 < iou < 0.7:
                # Check if below figure (typical caption position)
                if text_bbox[1] > figure_bbox[3]:  # text top > figure bottom
                    return "caption"
        
        return "separate"
    
    def _classify_with_layoutlm(
        self,
        text: str,
        bbox: List[float],
        page_image: np.ndarray
    ) -> str:
        """
        Classify using LayoutLMv3 model.
        Based on FINAL_TRY/Support-Code/ocr.py implementation.
        """
        # Convert image to PIL
        if isinstance(page_image, np.ndarray):
            page_image = Image.fromarray(page_image)
        
        width, height = page_image.size
        
        # Normalize bbox to 0-1000 range (LayoutLMv3 standard)
        normalized_box = [
            int(bbox[0] / width * 1000),
            int(bbox[1] / height * 1000),
            int(bbox[2] / width * 1000),
            int(bbox[3] / height * 1000)
        ]
        
        # Prepare input (following the pattern from ocr.py)
        encoding = self.processor(
            images=page_image,
            text=[text],
            boxes=[normalized_box],
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )
        
        # Run model
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1)
        
        # Map prediction to label
        # Note: This is a base model - for production, fine-tune on your dataset
        label_map = {
            0: "paragraph",
            1: "heading",
            2: "label",
            3: "caption"
        }
        
        pred_id = predictions[0][0].item()
        return label_map.get(pred_id, "paragraph")
    
    def _detect_relationship_with_layoutlm(
        self,
        text_region: Dict[str, Any],
        figure_region: Dict[str, Any],
        page_image: np.ndarray
    ) -> str:
        """
        Detect relationship using LayoutLMv3.
        Uses spatial context + model embeddings.
        """
        text_bbox = text_region.get('bbox')
        figure_bbox = figure_region.get('bbox')
        text = text_region.get('text', '')
        
        # Convert image to PIL
        if isinstance(page_image, np.ndarray):
            page_image = Image.fromarray(page_image)
        
        width, height = page_image.size
        
        # Normalize bboxes to 0-1000 range
        norm_text_box = [
            int(text_bbox[0] / width * 1000),
            int(text_bbox[1] / height * 1000),
            int(text_bbox[2] / width * 1000),
            int(text_bbox[3] / height * 1000)
        ]
        
        # Use LayoutLMv3 to analyze the text in context
        try:
            encoding = self.processor(
                images=page_image,
                text=[text],
                boxes=[norm_text_box],
                return_tensors="pt",
                truncation=True,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                # Use model's understanding of the text
                predictions = outputs.logits.argmax(-1)
                pred_id = predictions[0][0].item()
                
                # If model predicts caption, trust it
                if pred_id == 3:  # caption label
                    return "caption"
        except Exception as e:
            logger.debug(f"LayoutLMv3 relationship detection failed: {e}")
        
        # Fallback to spatial heuristics
        iou = self._compute_iou(text_bbox, figure_bbox)
        
        if iou > 0.7:
            return "inside_figure"
        elif iou > 0.1 and text_bbox[1] > figure_bbox[3]:
            return "caption"
        else:
            return "separate"
    
    def _classify_rule_based(self, text: str, font_info: Optional[Dict] = None) -> str:
        """Rule-based classification fallback."""
        if not text:
            return "paragraph"
        
        text_len = len(text.strip())
        
        # Very short text - likely label
        if text_len < 10:
            return "label"
        
        # Short text with font info
        if font_info:
            font_size = font_info.get('size', 12)
            if font_size > 14 and text_len < 100:
                return "heading"
        
        # Medium length - could be heading
        if text_len < 100:
            # Check if ends with punctuation (paragraph) or not (heading)
            if text.strip()[-1] in '.!?':
                return "paragraph"
            else:
                return "heading"
        
        # Long text - paragraph
        return "paragraph"
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
