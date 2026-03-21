"""
Table BBox Refiner - Snap table bounding boxes to word unions.

This module refines layout-detected table bboxes by:
1. Snapping to the tight union of overlapping words (stable anchor)
2. Optionally expanding to include nearby vector lines (for ruled tables)
"""

import logging
from typing import Optional, Tuple

from .types import BBoxPDF, TablePrimitives, DrawingPrimitive

logger = logging.getLogger(__name__)


class TableBboxRefiner:
    """
    Refine table bounding boxes by snapping to actual content.
    
    This prevents "drifting crops" that confuse TSR models and ensures
    consistent extraction across runs.
    """
    
    def __init__(
        self,
        word_overlap_threshold: float = 0.3,
        line_proximity_threshold: float = 5.0,  # Points
        margin: float = 2.0,  # Points
        min_line_length: float = 20.0,  # Minimum line length to consider
    ):
        """
        Args:
            word_overlap_threshold: Min overlap ratio to include a word
            line_proximity_threshold: Max distance to snap to a line
            margin: Margin to add after snapping (in points)
            min_line_length: Minimum line length to consider for expansion
        """
        self.word_overlap_threshold = word_overlap_threshold
        self.line_proximity_threshold = line_proximity_threshold
        self.margin = margin
        self.min_line_length = min_line_length
    
    def refine(
        self,
        initial_bbox: BBoxPDF,
        primitives: TablePrimitives,
        expand_to_lines: bool = True,
    ) -> Tuple[BBoxPDF, dict]:
        """
        Refine a table bounding box.
        
        Args:
            initial_bbox: Initial bbox from layout detection
            primitives: Extracted page primitives
            expand_to_lines: Whether to expand to nearby ruling lines
        
        Returns:
            Tuple of (refined_bbox, debug_info)
        """
        debug_info = {
            "initial_bbox": initial_bbox,
            "words_found": 0,
            "snap_bbox": None,
            "line_expansion": False,
            "final_bbox": None,
        }
        
        # Step 1: Find words overlapping with initial bbox
        words_in_bbox = primitives.get_words_in_bbox(
            initial_bbox, 
            overlap_threshold=self.word_overlap_threshold
        )
        debug_info["words_found"] = len(words_in_bbox)
        
        if not words_in_bbox:
            # No words found - return original bbox with margin
            logger.debug(f"No words found in bbox {initial_bbox}, returning with margin")
            final = self._add_margin(initial_bbox)
            debug_info["final_bbox"] = final
            return final, debug_info
        
        # Step 2: Compute tight union of word bboxes
        snap_bbox = self._compute_word_union(words_in_bbox)
        debug_info["snap_bbox"] = snap_bbox
        
        # Step 3: Optionally expand to include nearby ruling lines
        if expand_to_lines:
            ruled_score = self._compute_ruled_score(initial_bbox, primitives)
            if ruled_score > 0.1:  # Some line evidence
                snap_bbox = self._expand_to_lines(snap_bbox, primitives)
                debug_info["line_expansion"] = True
        
        # Step 4: Add margin
        final_bbox = self._add_margin(snap_bbox)
        debug_info["final_bbox"] = final_bbox
        
        logger.debug(
            f"Refined bbox: {initial_bbox} -> {final_bbox} "
            f"(words: {len(words_in_bbox)}, line_expand: {debug_info['line_expansion']})"
        )
        
        return final_bbox, debug_info
    
    def _compute_word_union(self, words) -> BBoxPDF:
        """Compute the tight bounding box containing all words."""
        if not words:
            return (0, 0, 0, 0)
        
        x0 = min(w.bbox[0] for w in words)
        y0 = min(w.bbox[1] for w in words)
        x1 = max(w.bbox[2] for w in words)
        y1 = max(w.bbox[3] for w in words)
        
        return (x0, y0, x1, y1)
    
    def _compute_ruled_score(
        self, 
        bbox: BBoxPDF, 
        primitives: TablePrimitives
    ) -> float:
        """
        Compute a score indicating how "ruled" a table region is.
        
        Score is based on density of horizontal/vertical lines.
        """
        drawings = primitives.get_drawings_in_bbox(bbox, overlap_threshold=0.3)
        
        if not drawings:
            return 0.0
        
        # Count and measure H/V lines
        h_line_length = 0.0
        v_line_length = 0.0
        
        for d in drawings:
            if d.kind == "line" and d.length >= self.min_line_length:
                if d.is_horizontal:
                    h_line_length += d.length
                elif d.is_vertical:
                    v_line_length += d.length
        
        # Normalize by bbox perimeter
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        perimeter = 2 * (bbox_width + bbox_height)
        
        if perimeter <= 0:
            return 0.0
        
        total_line_length = h_line_length + v_line_length
        return min(1.0, total_line_length / perimeter)
    
    def _expand_to_lines(
        self, 
        bbox: BBoxPDF, 
        primitives: TablePrimitives
    ) -> BBoxPDF:
        """
        Expand bbox to include nearby horizontal/vertical lines.
        
        This helps capture ruled table borders that may extend
        slightly beyond the text content.
        """
        x0, y0, x1, y1 = bbox
        
        # Get lines near the bbox
        search_bbox = (
            x0 - self.line_proximity_threshold * 3,
            y0 - self.line_proximity_threshold * 3,
            x1 + self.line_proximity_threshold * 3,
            y1 + self.line_proximity_threshold * 3,
        )
        nearby_drawings = primitives.get_drawings_in_bbox(search_bbox, overlap_threshold=0.1)
        
        for d in nearby_drawings:
            if d.kind != "line" or d.length < self.min_line_length:
                continue
            
            # Check if line is near bbox edge and should expand it
            if d.is_horizontal and len(d.points) >= 2:
                line_y = d.points[0][1]
                line_x0 = min(d.points[0][0], d.points[1][0])
                line_x1 = max(d.points[0][0], d.points[1][0])
                
                # Horizontal line near top edge
                if abs(line_y - y0) < self.line_proximity_threshold:
                    if line_x0 < x0 and line_x1 > x0:  # Line extends left
                        x0 = min(x0, line_x0)
                    if line_x1 > x1 and line_x0 < x1:  # Line extends right
                        x1 = max(x1, line_x1)
                    y0 = min(y0, line_y)
                
                # Horizontal line near bottom edge
                if abs(line_y - y1) < self.line_proximity_threshold:
                    if line_x0 < x0 and line_x1 > x0:
                        x0 = min(x0, line_x0)
                    if line_x1 > x1 and line_x0 < x1:
                        x1 = max(x1, line_x1)
                    y1 = max(y1, line_y)
            
            elif d.is_vertical and len(d.points) >= 2:
                line_x = d.points[0][0]
                line_y0 = min(d.points[0][1], d.points[1][1])
                line_y1 = max(d.points[0][1], d.points[1][1])
                
                # Vertical line near left edge
                if abs(line_x - x0) < self.line_proximity_threshold:
                    if line_y0 < y0 and line_y1 > y0:
                        y0 = min(y0, line_y0)
                    if line_y1 > y1 and line_y0 < y1:
                        y1 = max(y1, line_y1)
                    x0 = min(x0, line_x)
                
                # Vertical line near right edge
                if abs(line_x - x1) < self.line_proximity_threshold:
                    if line_y0 < y0 and line_y1 > y0:
                        y0 = min(y0, line_y0)
                    if line_y1 > y1 and line_y0 < y1:
                        y1 = max(y1, line_y1)
                    x1 = max(x1, line_x)
        
        return (x0, y0, x1, y1)
    
    def _add_margin(self, bbox: BBoxPDF) -> BBoxPDF:
        """Add margin to bbox."""
        return (
            bbox[0] - self.margin,
            bbox[1] - self.margin,
            bbox[2] + self.margin,
            bbox[3] + self.margin,
        )
