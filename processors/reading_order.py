"""
Reading Order Resolver - Determine correct reading order using adaptive algorithms
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for LayoutLMv3 availability
LAYOUTLM_AVAILABLE = False
try:
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    LAYOUTLM_AVAILABLE = True
except ImportError:
    pass

class ReadingOrderResolver:
    """Resolve reading order using Recursive XY-Cut"""
    
    def __init__(self, use_layoutlm: bool = False):
        """
        Initialize reading order resolver
        
        Args:
            use_layoutlm: Whether to use LayoutLMv3 if available
        """
        self.use_layoutlm = use_layoutlm and LAYOUTLM_AVAILABLE
        self.layoutlm_model = None
        self.layoutlm_processor = None
        
        if self.use_layoutlm:
            self._initialize_layoutlm()
    
    def _initialize_layoutlm(self):
        """Initialize LayoutLMv3 model for reading order"""
        try:
            logger.info("Initializing LayoutLMv3 for reading order...")
            # Note: This would require a fine-tuned model
            logger.info("LayoutLMv3 initialization skipped (requires fine-tuned model)")
            self.use_layoutlm = False
        except Exception as e:
            logger.warning(f"Failed to initialize LayoutLMv3: {e}")
            self.use_layoutlm = False
    
    def order_regions(
        self, 
        regions: List[Dict[str, Any]], 
        page_image: Optional[np.ndarray] = None,
        doc_profile: Optional[Any] = None,
        strategy_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Order regions using Recursive XY-Cut algorithm.
        This is a robust structural analysis method that works for any Manhattan layout.
        
        Args:
            regions: List of detected regions with bboxes
            page_image: Unused in XY-Cut (uses bbox coordinates)
            doc_profile: Optional profile to tune gap thresholds
            strategy_override: Optional strategy suggested by VLM Planner
            
        Returns:
            Ordered list of regions
        """
        if not regions:
            return []
            
        # Filter out regions without bboxes
        valid_regions = [r for r in regions if r.get('bbox')]
        if not valid_regions:
            return regions
            
        # Recursive XY Cost Parameters
        self.min_x_gap = 10  # Reduced: 10px gap for columns
        self.min_y_gap = 2   # Reduced: 2px gap for rows/headers
        
        # Start recursion
        ordered_indices = self._recursive_xy_cut(valid_regions, strategy_override=strategy_override)
        
        # Reconstruct ordered list
        result = [valid_regions[i] for i in ordered_indices]
        return result

    def _recursive_xy_cut(self, regions: List[Dict[str, Any]], depth: int = 0, strategy_override: Optional[str] = None) -> List[int]:
        """
        Recursively cut regions into reading order. 
        Returns list of INDICES relative to the input list.
        """
        indent = "  " * depth
        if len(regions) <= 1:
            return list(range(len(regions)))
            
        # 1. Calculate Projection Gaps
        
        # X-Axis (Columns)
        x_intervals = [(r['bbox'][0], r['bbox'][2], i) for i, r in enumerate(regions)]
        best_x_gap, split_x = self._find_widest_gap(x_intervals, self.min_x_gap)
        
        # Y-Axis (Rows/Sections)
        y_intervals = [(r['bbox'][1], r['bbox'][3], i) for i, r in enumerate(regions)]
        best_y_gap, split_y = self._find_widest_gap(y_intervals, self.min_y_gap)
        
        # Debug Log
        # try:
        #      with open("debug_rxyc.log", "a") as f:
        #          f.write(f"{indent}Depth {depth}: len={len(regions)}. X-Gap={best_x_gap} (at {split_x}), Y-Gap={best_y_gap} (at {split_y})\\n")
        # except: pass

        use_x = False
        use_y = False
        
        # Decision Logic:
        # If both cuts are possible:
        # - If X-gap is massive (>30), it's likely columns. Take it.
        # - Otherwise prefer Y-gap to strip headers/footers first.
        # This handles the Header Blocking Column Split case:
        #   Header -> X-gap=0 -> Only Y-gap available -> Use Y.
        
        if split_x and split_y:
            # If VLM suggests column-first, prioritize X-cut even if Y-gap is available
            if strategy_override == "xy_cut_column_first" and best_x_gap > 15:
                use_x = True
            elif best_x_gap > 30: 
                use_x = True
            else:
                use_y = True
        elif split_x:
            use_x = True
        elif split_y:
            use_y = True
            
        if use_x:
            # Split Vertically (Left/Right)
            left_group = []
            right_group = []
            for i, r in enumerate(regions):
                center = (r['bbox'][0] + r['bbox'][2]) / 2
                if center < split_x:
                    left_group.append((r, i))
                else:
                    right_group.append((r, i))
            
            # Recurse
            left_indices = self._recursive_xy_cut([item[0] for item in left_group], depth+1, strategy_override=strategy_override)
            right_indices = self._recursive_xy_cut([item[0] for item in right_group], depth+1, strategy_override=strategy_override)
            
            # Combine indices
            combined = [left_group[i][1] for i in left_indices] + [right_group[i][1] for i in right_indices]
            return combined

        if use_y:
            # Split Horizontally (Top/Bottom)
            top_group = []
            bottom_group = []
            for i, r in enumerate(regions):
                center = (r['bbox'][1] + r['bbox'][3]) / 2
                if center < split_y:
                    top_group.append((r, i))
                else:
                    bottom_group.append((r, i))
            
            top_indices = self._recursive_xy_cut([item[0] for item in top_group], depth+1, strategy_override=strategy_override)
            bottom_indices = self._recursive_xy_cut([item[0] for item in bottom_group], depth+1, strategy_override=strategy_override)
            
            combined = [top_group[i][1] for i in top_indices] + [bottom_group[i][1] for i in bottom_indices]
            return combined
            
        # Leaf Node (Block): Sort by Y then X
        indexed_regions = list(enumerate(regions))
        indexed_regions.sort(key=lambda item: (item[1]['bbox'][1], item[1]['bbox'][0]))
        return [item[0] for item in indexed_regions]

    def _find_widest_gap(self, intervals: List[Tuple[float, float, int]], min_gap: float) -> Tuple[float, Optional[float]]:
        """
        Find the split point of the widest gap in projection profile.
        Returns: (gap_width, split_center)
        """
        intervals.sort(key=lambda x: x[0])
        if not intervals:
            return 0, None
            
        merged = []
        curr_start, curr_end, _ = intervals[0]
        for next_start, next_end, _ in intervals[1:]:
            if next_start < curr_end:
                # Overlap
                curr_end = max(curr_end, next_end)
            else:
                # Disjoint: We found a gap!
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
        
        # If only 1 merged interval, there are no gaps
        if len(merged) < 2:
            return 0, None
            
        max_gap_width = 0
        best_split = None
        
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i+1][0]
            gap_width = gap_end - gap_start
            
            if gap_width > max_gap_width and gap_width >= min_gap:
                max_gap_width = gap_width
                best_split = (gap_start + gap_end) / 2
                
        return max_gap_width, best_split
