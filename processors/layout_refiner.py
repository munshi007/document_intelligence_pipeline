"""
SOTA Layout Refining Agent.
Ensures 'Absolute Best' research-worthy performance by:
1. Identifying gaps in layout detection (Visual Gap Analysis).
2. Refining coordinates to sub-pixel character boundaries (Sub-Pixel Refinement).
3. Ensembling VLM insights with YOLO/LayoutParser detections.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from common.vlm_client import VLMClient
from common.vlm_types import VisualGapAnalysis, RefinedRegion
from common.types import LayoutRegion, BBox

logger = logging.getLogger(__name__)

class LayoutRefiningAgent:
    """Agentic layer for layout precision and recovery."""

    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def refine_layout_ensemble(
        self, 
        image: Image.Image, 
        existing_regions: List[LayoutRegion],
        page_num: int,
        metadata: Optional[Dict] = None
    ) -> List[LayoutRegion]:
        """
        SOTA: Refine the existing ensemble by scanning for missed content.
        """
        logger.info(f"LayoutRefiningAgent: Analyzing ensemble coverage ({len(existing_regions)} regions)...")
        
        # 1. Perform Visual Gap Analysis on the whole page
        # Using the SOTA resolution (1600px) via is_complex=True
        prompt = (
            "Analyze this document page. "
            "I have already detected some regions, but I might have missed small floating text, captions, or logos. "
            "Identify any CONTENT (text, tables, figures) that is NOT well-captured by a bounding box. "
            "Provide the coordinates in normalized [ymin, xmin, ymax, xmax] (0-1000) format."
        )
        
        gap_analysis = self.vlm_client.generate_structured(
            image=image,
            prompt=prompt,
            response_model=VisualGapAnalysis,
            is_complex=True,
            metadata=metadata
        )
        
        if not gap_analysis or not gap_analysis.found_missed_content:
            logger.info("LayoutRefiningAgent: No missed content found by SOTA scan.")
            return existing_regions
            
        logger.info(f"LayoutRefiningAgent: SOTA scan found {len(gap_analysis.missed_regions)} missed regions!")
        
        # 2. Convert RefinedRegion to LayoutRegion and merge
        refined_list = existing_regions.copy()
        img_w, img_h = image.size
        
        for missed in gap_analysis.missed_regions:
            # Convert normalized 1000x1000 to pixel coordinates
            ymin, xmin, ymax, xmax = missed.refined_bbox
            pixel_bbox = [
                xmin * img_w / 1000,
                ymin * img_h / 1000,
                xmax * img_w / 1000,
                ymax * img_h / 1000
            ]
            
            new_region = LayoutRegion(
                region_id=f"vlm_refiner_{page_num}_{len(refined_list)}",
                page_num=page_num,
                bbox=BBox(
                    x0=pixel_bbox[0],
                    y0=pixel_bbox[1],
                    x1=pixel_bbox[2],
                    y1=pixel_bbox[3]
                ),
                type=missed.label.title(),
                confidence=missed.confidence,
                source="vlm_refiner"
            )
            refined_list.append(new_region)
            
        return refined_list

    def sub_pixel_snap(self, image: Image.Image, region: LayoutRegion) -> LayoutRegion:
        """
        Refines a single fuzzy bounding box to exact content boundaries.
        Useful for complex tables or dense technical text.
        """
        # Crop the region with 20% padding
        bbox = region.bbox
        w = bbox.x1 - bbox.x0
        h = bbox.y1 - bbox.y0
        
        pad_x = w * 0.2
        pad_y = h * 0.2
        
        crop_bbox = [
            max(0, bbox.x0 - pad_x),
            max(0, bbox.y0 - pad_y),
            min(image.width, bbox.x1 + pad_x),
            min(image.height, bbox.y1 + pad_y)
        ]
        
        crop_img = image.crop(crop_bbox)
        
        prompt = (
            f"This is a crop of a '{region.label}' region. "
            "The current bounding box is approximate. Please provide the EXACT content boundaries "
            "so that no text is cut off and there is minimal white space. "
            "Output normalized coordinates relative to THIS CROP (0-1000)."
        )
        
        refined = self.vlm_client.generate_structured(
            image=crop_img,
            prompt=prompt,
            response_model=RefinedRegion,
            is_complex=True
        )
        
        if not refined:
            return region
            
        # Transform back to page coordinates
        ymin, xmin, ymax, xmax = refined.refined_bbox
        crop_w = crop_bbox[2] - crop_bbox[0]
        crop_h = crop_bbox[3] - crop_bbox[1]
        
        final_bbox = [
            crop_bbox[0] + (xmin * crop_w / 1000),
            crop_bbox[1] + (ymin * crop_h / 1000),
            crop_bbox[0] + (xmax * crop_w / 1000),
            crop_bbox[1] + (ymax * crop_h / 1000)
        ]
        
        region.bbox = BBox(x0=final_bbox[0], y0=final_bbox[1], x1=final_bbox[2], y1=final_bbox[3])
        region.label = refined.label.title()
        region.confidence = (region.confidence + refined.confidence) / 2
        
        return region
