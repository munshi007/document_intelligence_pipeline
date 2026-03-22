"""
Figure and Caption Processor - Deterministic Association for RT-DETR
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FigureCaptionProcessor:
    """
    Associate figures and tables with captions using RT-DETR detections and spatial analysis.
    """
    
    def __init__(self):
        """Initialize figure-caption processor."""
        # Safety patterns for native text orphans
        self.caption_patterns = [
            r'^Figure\s+\d+', r'^Fig\.\s*\d+', r'^Table\s+\d+',
            r'^Diagram\s+\d+', r'^Chart\s+\d+'
        ]
    
    def associate_captions(
        self,
        regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Associate figure/table regions with captions via proximity.
        """
        if not regions:
            return regions
        
        # 1. Separate target regions (RT-DETR labels)
        figures = [r for r in regions if r.get('type') == 'Figure']
        tables = [r for r in regions if r.get('type') == 'Table']
        captions = [r for r in regions if r.get('type') == 'Caption']
        
        logger.info(f"Caption Processor: Found {len(figures)} figs, {len(tables)} tables, {len(captions)} captions")
        
        # 2. Associate generic 'Caption' labels with nearest Figure/Table
        for caption in captions:
            parent = self._find_nearest_parent(caption, figures + tables)
            if parent:
                parent['caption'] = (parent.get('caption', '') + " " + (caption.get('text', ''))).strip()
                parent['caption_bbox'] = caption.get('bbox')
                caption['associated_with'] = parent.get('region_id')

        # 3. Fallback: Check 'Text' regions for missed caption patterns
        text_regions = [r for r in regions if r.get('type') == 'Text' and not r.get('associated_with')]
        for tr in text_regions:
            text = (tr.get('text') or '').strip()
            if self._matches_pattern(text):
                parent = self._find_nearest_parent(tr, figures + tables)
                if parent and not parent.get('caption'):
                    parent['caption'] = text
                    parent['caption_bbox'] = tr.get('bbox')
                    tr['type'] = 'Caption'
                    tr['associated_with'] = parent.get('region_id')
        
        return regions
    
    def _find_nearest_parent(self, caption: Dict[str, Any], parents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest Figure/Table for a given caption."""
        if not parents: return None
        
        c_bbox = caption['bbox']
        c_center_x = (c_bbox[0] + c_bbox[2]) / 2
        
        best_dist = float('inf')
        best_parent = None
        
        for p in parents:
            p_bbox = p['bbox']
            p_center_x = (p_bbox[0] + p_bbox[2]) / 2
            
            # Vertical proximity (Caption usually below or just above)
            # Distance from caption-top to parent-bottom (below) or parent-top to caption-bottom (above)
            dist_below = abs(c_bbox[1] - p_bbox[3])
            dist_above = abs(p_bbox[1] - c_bbox[3])
            vertical_dist = min(dist_below, dist_above)
            
            # Horizontal alignment
            h_overlap = max(0, min(c_bbox[2], p_bbox[2]) - max(c_bbox[0], p_bbox[0]))
            h_dist = abs(c_center_x - p_center_x)
            
            # Heuristic: Captions should be close vertically and somewhat aligned horizontally
            if vertical_dist < 60 and (h_overlap > 0 or h_dist < 150):
                if vertical_dist < best_dist:
                    best_dist = vertical_dist
                    best_parent = p
                    
        return best_parent

    def _matches_pattern(self, text: str) -> bool:
        """Check if text matches standard caption naming conventions."""
        for pattern in self.caption_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
