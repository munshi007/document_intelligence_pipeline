"""
Table Type Router - Classify tables as RULED, KV, or COMPLEX.

Uses data-driven scoring (not arbitrary thresholds) with ordered decision logic:
1. RULED: High vector line density
2. KV: Strong 2-column separability
3. COMPLEX: Everything else (TSR fallback)
"""

import logging
from typing import Dict, Tuple, Optional, List
import math

from .types import BBoxPDF, TablePrimitives, TableType, WordSpan

logger = logging.getLogger(__name__)


class TableTypeRouter:
    """
    Route tables to the appropriate extraction strategy.
    
    Uses ordered thresholds (not argmax) for stability:
    - If ruled_score > R → RULED
    - Elif kv_score > K → KV  
    - Else → COMPLEX
    
    Thresholds can be adjusted by VLM planner priors.
    """
    
    def __init__(
        self,
        ruled_threshold: float = 0.3,
        kv_threshold: float = 0.4,
        min_words_for_analysis: int = 4,
        min_line_length: float = 15.0,
    ):
        """
        Args:
            ruled_threshold: Minimum ruled_score to classify as RULED
            kv_threshold: Minimum kv_score to classify as KV
            min_words_for_analysis: Minimum words needed for meaningful analysis
            min_line_length: Minimum line length (points) to consider
        """
        self.ruled_threshold = ruled_threshold
        self.kv_threshold = kv_threshold
        self.min_words_for_analysis = min_words_for_analysis
        self.min_line_length = min_line_length
    
    def route(
        self,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        priors: Optional[Dict[str, float]] = None,
    ) -> Tuple[TableType, Dict[str, float]]:
        """
        Determine the table type for extraction.
        
        Args:
            bbox: Refined table bounding box
            primitives: Page primitives
            priors: Optional VLM priors to adjust thresholds
        
        Returns:
            Tuple of (TableType, scores_dict)
        """
        priors = priors or {}
        
        # Get words and drawings in bbox
        words = primitives.get_words_in_bbox(bbox, overlap_threshold=0.5)
        drawings = primitives.get_drawings_in_bbox(bbox, overlap_threshold=0.3)
        
        # Compute scores
        ruled_score = self._compute_ruled_score(bbox, drawings)
        kv_score = self._compute_kv_score(bbox, words)
        
        scores = {
            "ruled_score": ruled_score,
            "kv_score": kv_score,
            "word_count": len(words),
            "drawing_count": len(drawings),
        }
        
        # Adjust thresholds based on priors
        r_threshold = self.ruled_threshold
        k_threshold = self.kv_threshold
        
        if priors.get("ruled_likelihood", 0) > 0.7:
            r_threshold *= 0.7  # Lower threshold if VLM thinks it's ruled
        if priors.get("kv_likelihood", 0) > 0.7:
            k_threshold *= 0.7
        
        # Ordered decision logic (prevents flip-flopping)
        if len(words) < self.min_words_for_analysis:
            print(f"[router] Too few words ({len(words)}), defaulting to COMPLEX")
            return TableType.COMPLEX, scores
        
        if ruled_score >= r_threshold:
            print(f"[router] → RULED: ruled_score={ruled_score:.2f} >= {r_threshold:.2f}")
            return TableType.RULED, scores
        
        if kv_score >= k_threshold:
            print(f"[router] → KV: kv_score={kv_score:.2f} >= {k_threshold:.2f}")
            return TableType.KV, scores
        
        print(f"[router] → COMPLEX: ruled={ruled_score:.2f}, kv={kv_score:.2f}")
        return TableType.COMPLEX, scores
    
    def _compute_ruled_score(
        self, 
        bbox: BBoxPDF, 
        drawings: List
    ) -> float:
        """
        Compute ruled table score based on vector line density.
        
        Score = (total H/V line length) / (bbox perimeter)
        Normalized to [0, 1] with saturation at 1.0
        """
        if not drawings:
            return 0.0
        
        h_line_length = 0.0
        v_line_length = 0.0
        h_line_count = 0
        v_line_count = 0
        
        for d in drawings:
            if d.kind == "line" and d.length >= self.min_line_length:
                if d.is_horizontal:
                    h_line_length += d.length
                    h_line_count += 1
                elif d.is_vertical:
                    v_line_length += d.length
                    v_line_count += 1
        
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        perimeter = 2 * (bbox_width + bbox_height)
        
        if perimeter <= 0:
            return 0.0
        
        # Need both horizontal AND vertical lines for a proper grid
        if h_line_count < 2 or v_line_count < 2:
            return 0.0
        
        total_line_length = h_line_length + v_line_length
        score = total_line_length / perimeter
        
        return min(1.0, score)
    
    def _compute_kv_score(
        self, 
        bbox: BBoxPDF, 
        words: List[WordSpan]
    ) -> float:
        """
        Compute KV (key-value / 2-column) score based on X-clustering.
        
        Uses the "max gap ratio" method:
        1. Compute x-centers of all words
        2. Sort and find gaps between adjacent words
        3. If there's one dominant gap in the middle, it's likely 2-column
        
        Score = max_gap / total_width, with bonuses for row consistency
        """
        if len(words) < 4:
            return 0.0
        
        bbox_width = bbox[2] - bbox[0]
        if bbox_width <= 0:
            return 0.0
        
        # Get x-centers of words
        x_centers = [(w.bbox[0] + w.bbox[2]) / 2 for w in words]
        x_centers.sort()
        
        if len(x_centers) < 2:
            return 0.0
        
        # Find gaps between adjacent x-centers
        gaps = []
        for i in range(len(x_centers) - 1):
            gap = x_centers[i + 1] - x_centers[i]
            gaps.append((gap, (x_centers[i] + x_centers[i + 1]) / 2))
        
        if not gaps:
            return 0.0
        
        # Find the maximum gap
        max_gap, gap_position = max(gaps, key=lambda x: x[0])
        
        # Check if the gap is roughly in the middle third of the bbox
        bbox_center = (bbox[0] + bbox[2]) / 2
        gap_relative_pos = (gap_position - bbox[0]) / bbox_width
        
        # The gap should be roughly between 0.25 and 0.75 of the width
        if not (0.2 < gap_relative_pos < 0.8):
            return 0.0
        
        # Base score: gap ratio
        gap_ratio = max_gap / bbox_width
        
        # Bonus for row consistency (words tend to be on left or right)
        left_count = sum(1 for x in x_centers if x < gap_position)
        right_count = sum(1 for x in x_centers if x > gap_position)
        
        # Ideally, roughly equal split
        balance = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0
        
        # Combined score
        score = gap_ratio * 0.7 + balance * 0.3
        
        return min(1.0, score * 2)  # Scale up for threshold comparison
    
    def _compute_row_structure_bonus(
        self, 
        words: List[WordSpan],
        gap_position: float
    ) -> float:
        """
        Compute a bonus for tables with consistent row structure.
        
        Groups words by Y-position and checks if rows have words on both sides.
        """
        if not words:
            return 0.0
        
        # Group words by approximate Y position (row)
        y_tolerance = self._estimate_row_tolerance(words)
        rows = self._group_by_y(words, y_tolerance)
        
        if len(rows) < 2:
            return 0.0
        
        # Check how many rows have words on both sides of the gap
        consistent_rows = 0
        for row_words in rows:
            has_left = any((w.bbox[0] + w.bbox[2]) / 2 < gap_position for w in row_words)
            has_right = any((w.bbox[0] + w.bbox[2]) / 2 > gap_position for w in row_words)
            if has_left and has_right:
                consistent_rows += 1
        
        return consistent_rows / len(rows)
    
    def _estimate_row_tolerance(self, words: List[WordSpan]) -> float:
        """Estimate row grouping tolerance based on median font size."""
        if not words:
            return 10.0
        
        font_sizes = [w.font_size for w in words if w.font_size > 0]
        if font_sizes:
            median_size = sorted(font_sizes)[len(font_sizes) // 2]
            return max(5.0, median_size * 0.6)
        
        # Fallback: estimate from y-differences
        y_coords = sorted(set((w.bbox[1] + w.bbox[3]) / 2 for w in words))
        if len(y_coords) > 1:
            diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            if diffs:
                return max(5.0, min(diffs) * 0.8)
        
        return 10.0
    
    def _group_by_y(
        self, 
        words: List[WordSpan], 
        tolerance: float
    ) -> List[List[WordSpan]]:
        """Group words into rows by Y-coordinate."""
        if not words:
            return []
        
        sorted_words = sorted(words, key=lambda w: w.bbox[1])
        rows = []
        current_row = [sorted_words[0]]
        current_y = sorted_words[0].bbox[1]
        
        for word in sorted_words[1:]:
            if abs(word.bbox[1] - current_y) <= tolerance:
                current_row.append(word)
            else:
                rows.append(current_row)
                current_row = [word]
                current_y = word.bbox[1]
        
        if current_row:
            rows.append(current_row)
        
        return rows
