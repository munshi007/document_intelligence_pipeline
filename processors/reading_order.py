"""
Reading Order Resolver - Pure Recursive XY-Cut
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ReadingOrderResolver:
    """Resolve reading order using a deterministic Recursive XY-Cut algorithm."""
    
    def __init__(self, **kwargs):
        """Initialize reading order resolver."""
        # Standardize on XY-Cut for production
        self.min_x_gap = 10  # Minimum gap to consider a column split
        self.min_y_gap = 2   # Minimum gap to consider a horizontal split
        logger.info("ReadingOrderResolver: Initialized with Pure XY-Cut strategy.")
    
    def order_regions(
        self, 
        regions: List[Dict[str, Any]], 
        page_image: Optional[Any] = None,
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Order regions using Recursive XY-Cut.
        """
        if not regions:
            return []
            
        valid_regions = [r for r in regions if r.get('bbox')]
        if not valid_regions:
            return regions
            
        # Recursive processing
        ordered_indices = self._recursive_xy_cut(valid_regions)
        return [valid_regions[i] for i in ordered_indices]

    def _recursive_xy_cut(self, regions: List[Dict[str, Any]], depth: int = 0) -> List[int]:
        """
        Recursively cut regions into reading order. Returns list of original indices.
        """
        if len(regions) <= 1:
            return [0]
            
        # 1. Find widest gaps in X and Y projection
        best_x_gap, split_x = self._find_widest_split(regions, axis=0, min_gap=self.min_x_gap)
        best_y_gap, split_y = self._find_widest_split(regions, axis=1, min_gap=self.min_y_gap)
        
        # 2. Decision Logic
        # If X-gap is significant (>30), it's likely a multi-column layout. Split X first.
        # Otherwise, split Y (rows) first to maintain flow.
        use_x = False
        if split_x and split_y:
            use_x = (best_x_gap > 35) # Heuristic for clear columns
        elif split_x:
            use_x = True
        elif split_y:
            use_x = False
        else:
            # Leaf node: Sort by position (Y then X)
            indexed = sorted(enumerate(regions), key=lambda x: (x[1]['bbox'][1], x[1]['bbox'][0]))
            return [i for i, _ in indexed]

        # 3. Perform Split
        side_a, side_b = [], []
        axis_idx = 0 if use_x else 1
        split_point = split_x if use_x else split_y
        
        for i, r in enumerate(regions):
            center = (r['bbox'][axis_idx] + r['bbox'][axis_idx+2]) / 2
            if center < split_point:
                side_a.append((r, i))
            else:
                side_b.append((r, i))
        
        # 4. Recurse and Map Indices
        res_a = self._recursive_xy_cut([x[0] for x in side_a], depth + 1)
        res_b = self._recursive_xy_cut([x[0] for x in side_b], depth + 1)
        
        return [side_a[i][1] for i in res_a] + [side_b[i][1] for i in res_b]

    def _find_widest_split(self, regions: List[Dict[str, Any]], axis: int, min_gap: float):
        """Find the widest projection gap on a given axis."""
        # axis=0: X, axis=1: Y
        intervals = sorted([(r['bbox'][axis], r['bbox'][axis+2]) for r in regions], key=lambda x: x[0])
        
        # Merge overlapping intervals
        merged = []
        curr_start, curr_end = intervals[0]
        for ns, ne in intervals[1:]:
            if ns < curr_end: curr_end = max(curr_end, ne)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = ns, ne
        merged.append((curr_start, curr_end))
        
        if len(merged) < 2: return 0, None
        
        best_gap, split_point = 0, None
        for i in range(len(merged) - 1):
            gap = merged[i+1][0] - merged[i][1]
            if gap > best_gap and gap >= min_gap:
                best_gap = gap
                split_point = (merged[i][1] + merged[i+1][0]) / 2
                
        return best_gap, split_point
