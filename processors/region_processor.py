"""
Region Processor - Deterministic Hierarchical Processing
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Configurable thresholds
OVERLAP_THRESHOLDS = {
    'figure': 0.75,
    'table': 0.4,
    'default': 0.5
}

class RegionProcessor:
    """
    Process regions hierarchically, trusting layout model's semantic labels.
    Filters out text that's inside figures/tables via pure CV.
    """
    
    def __init__(self, use_layoutlm: bool = False):
        """Initialize region processor."""
        # LayoutLM is deprecated in this production pipeline
        self.use_layoutlm = False
        logger.info("RegionProcessor: Initialized in Deterministic-Only mode.")
    
    def process_regions_hierarchically(
        self,
        layout_regions: List[Dict[str, Any]],
        text_regions: List[Dict[str, Any]],
        page_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Process regions in priority order, filtering text inside figures/tables.
        """
        # Separate layout regions by type
        figures = [r for r in layout_regions if r.get('type') == 'Figure']
        tables = [r for r in layout_regions if r.get('type') == 'Table']
        captions = [r for r in layout_regions if 'Caption' in r.get('type', '')]
        others = [r for r in layout_regions if r.get('type') not in ['Figure', 'Table'] and 'Caption' not in r.get('type', '')]
        
        # Filter native text regions that are inside occupied space (Figures/Tables)
        occupied_bboxes = figures + tables
        filtered_native_text = self._filter_text_inside_regions(text_regions, occupied_bboxes)
        
        # Combine in priority order
        processed_regions = []
        processed_regions.extend(figures)
        processed_regions.extend(tables)
        processed_regions.extend(captions)
        processed_regions.extend(others)
        processed_regions.extend(filtered_native_text)
        
        logger.info(f"Hierarchical processing complete: {len(processed_regions)} regions total.")
        return processed_regions

    def _filter_text_inside_regions(
        self,
        text_regions: List[Dict[str, Any]],
        occupied_regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out text regions that significant overlap with high-priority occupied regions.
        """
        if not occupied_regions:
            return text_regions
        
        filtered = []
        for text_region in text_regions:
            text_bbox = text_region.get('bbox')
            if not text_bbox: continue
            
            text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
            if text_area <= 0: continue
            
            is_inside = False
            total_intersection_area = 0.0
            
            for occupied in occupied_regions:
                occupied_bbox = occupied.get('bbox')
                if not occupied_bbox: continue
                
                # Calculate intersection
                x1_min, y1_min, x1_max, y1_max = text_bbox
                x2_min, y2_min, x2_max, y2_max = occupied_bbox
                
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                
                if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    
                    # Individual threshold check
                    region_type = occupied.get('type', '').lower()
                    threshold = OVERLAP_THRESHOLDS.get(region_type, OVERLAP_THRESHOLDS['default'])
                    
                    if (inter_area / text_area) > threshold:
                        is_inside = True
                        break
                    
                    total_intersection_area += inter_area
            
            # Final sum check
            if not is_inside and (total_intersection_area / text_area) > OVERLAP_THRESHOLDS['default']:
                is_inside = True
                
            if not is_inside:
                filtered.append(text_region)
        
        return filtered
