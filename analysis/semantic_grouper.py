"""
Semantic Text Grouper - Group text by spatial proximity and alignment (Geometric Fallback)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SemanticTextGrouper:
    """Group text regions using strict geometric rules to preserve structural integrity"""
    
    def __init__(self):
        """Initialize semantic text grouper"""
        pass
    
    def group_paragraphs(
        self,
        text_regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None,
        weights: Any = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Group text regions into paragraphs using robust geometric line merging.
        
        Args:
            text_regions: List of text regions
            doc_profile: Optional DocumentProfile
            weights: Ignored (legacy compatibility)
            
        Returns:
            List of paragraph groups
        """
        if not text_regions:
            return []
        
        # Filter valid regions
        valid_regions = [r for r in text_regions if r.get('text', '').strip()]
        if not valid_regions:
            return []
            
        # Use geometric grouping
        groups = self._geometric_grouping(valid_regions)
        
        logger.info(f"Grouped {len(valid_regions)} text lines into {len(groups)} paragraphs")
        return groups

    def _geometric_grouping(self, regions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group lines based on vertical proximity and horizontal alignment.
        Strictly respects column boundaries.
        """
        if not regions:
            return []
            
        # Add index to regions for tracking
        for i, r in enumerate(regions):
            r['_original_index'] = i
            r['_visited'] = False
            
        # Sort by Y-coordinate primarily
        regions_sorted = sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        groups = []
        
        for i, current in enumerate(regions_sorted):
            if current.get('_visited'):
                continue
                
            # Start a new group
            current_group = [current]
            current['_visited'] = True
            
            # Estimate line height (use current region height as baseline)
            line_height = current['bbox'][3] - current['bbox'][1]
            max_vertical_gap = line_height * 1.5  # Max gap between lines in a paragraph
            
            # Find neighbors
            # Look ahead in the sorted list
            for j in range(i + 1, len(regions_sorted)):
                candidate = regions_sorted[j]
                
                if candidate.get('_visited'):
                    continue
                
                # Check vertical distance (gap between bottom of current and top of candidate)
                # Since we are building incrementally, comp to the LAST added line in group
                last_line = current_group[-1]
                vertical_gap = candidate['bbox'][1] - last_line['bbox'][3]
                
                # If gap is too large (new paragraph or different section), stop checking this branch
                # But wait, sorting is by Y, so widely separated Y means we stop.
                if vertical_gap > max_vertical_gap:
                    # Could be end of paragraph, but check if it's just a slight gap.
                    # If > 3x line height, definitely stop.
                     if vertical_gap > line_height * 3:
                         break
                     continue # Too far to be next line, but maybe another line fits? No, sorted by Y.

                # Determine if lines are in the same column
                if self._is_same_column(last_line, candidate):
                    # Check vertical proximity strictly
                    if vertical_gap <= max_vertical_gap and vertical_gap >= -line_height * 0.5:
                         # Merge
                         current_group.append(candidate)
                         candidate['_visited'] = True
            
            groups.append(current_group)
            
        return groups

    def _is_same_column(self, r1: Dict, r2: Dict) -> bool:
        """Check if two regions presumably belong to the same column (horizontal overlap or alignment)."""
        b1 = r1['bbox']
        b2 = r2['bbox']
        
        # Horizontal overlap
        x_min = max(b1[0], b2[0])
        x_max = min(b1[2], b2[2])
        overlap = max(0, x_max - x_min)
        
        width1 = b1[2] - b1[0]
        width2 = b2[2] - b2[0]
        min_width = min(width1, width2)
        
        # If significant overlap (>50% of smaller width), likely same column
        if min_width > 0 and (overlap / min_width) > 0.5:
            return True
            
        # Left alignment check (within tolerance)
        if abs(b1[0] - b2[0]) < 20: # 20px tolerance
            # Also need to ensure they aren't miles apart horizontally (visual check)
            # But the overlap check covers most cases. 
            # If they are left aligned but no overlap? (e.g. really short line followed by long line)
            # Check if one is contained in the x-range of the other
            if (b1[0] >= b2[0] and b1[2] <= b2[2]) or (b2[0] >= b1[0] and b2[2] <= b1[2]):
                return True
            return True
            
        return False
        
    def detect_list_structures(self, text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect list structures using pattern recognition."""
        import re
        bullet_pattern = r'^\s*[•\-\*\+◦▪▫]\s'
        numbered_pattern = r'^\s*\d+[\.\)]\s'
        
        for region in text_regions:
            text = region.get('text', '')
            if re.match(bullet_pattern, text):
                region['is_list_item'] = True
                region['list_type'] = 'bullet'
            elif re.match(numbered_pattern, text):
                region['is_list_item'] = True
                region['list_type'] = 'numbered'
            else:
                region['is_list_item'] = False
        return text_regions
        
    def detect_footnotes_and_captions(self, text_regions: List[Dict], doc_profile=None) -> List[Dict]:
        """Detect footnotes/captions based on rules."""
        import re
        caption_pattern = r'^(Figure|Fig\.|Table|Diagram)\s+\d+'
        
        if not text_regions: return text_regions
        max_y = max((r['bbox'][3] for r in text_regions), default=1000)
        
        for r in text_regions:
            text = r.get('text', '')
            is_bottom = r['bbox'][1] > (max_y * 0.9)
            font_size = r.get('font_size', 12)
            
            if is_bottom and font_size < 10:
                r['is_footnote'] = True
            else:
                r['is_footnote'] = False
                
            if re.match(caption_pattern, text.strip(), re.IGNORECASE):
                r['is_caption'] = True
            else:
                r['is_caption'] = False
                
        return text_regions

