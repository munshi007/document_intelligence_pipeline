"""
Table QA - Quality Assurance metrics for table extraction.

Evaluates extracted tables using:
1. Coverage: % of input words assigned to cells
2. Duplication: Words used in multiple cells
3. Row Sanity: Consistent column count per row
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from .types import BBoxPDF, TablePrimitives, TableResult, TableQAMetrics

logger = logging.getLogger(__name__)


class TableQA:
    """
    Quality assurance for extracted tables.
    
    Computes metrics and determines if extraction passed.
    """
    
    def __init__(
        self,
        min_coverage: float = 0.70,
        max_duplication: float = 0.05,
        require_sanity: bool = True,
    ):
        """
        Args:
            min_coverage: Minimum word coverage to pass (0-1)
            max_duplication: Maximum allowed duplication ratio (0-1)
            require_sanity: Require consistent column count
        """
        self.min_coverage = min_coverage
        self.max_duplication = max_duplication
        self.require_sanity = require_sanity
    
    def evaluate(
        self,
        result: TableResult,
        primitives: TablePrimitives,
        table_bbox: BBoxPDF,
    ) -> TableQAMetrics:
        """
        Evaluate table extraction quality.
        
        Args:
            result: Extraction result
            primitives: Page primitives
            table_bbox: Table bounding box
        
        Returns:
            TableQAMetrics with scores and pass/fail
        """
        # Get words in table bbox
        words_in_bbox = self._get_words_in_bbox(primitives, table_bbox)
        total_words = len(words_in_bbox)
        
        if total_words == 0:
            return TableQAMetrics(
                coverage=1.0,
                duplication_ratio=0.0,
                row_sanity_score=1.0,
                empty_cell_ratio=0.0,
                passed=True,
                failure_reasons=[],
                unassigned_word_ids=[],
            )
        
        # Collect assigned word IDs
        assigned_ids: Set[str] = set()
        id_counts: Dict[str, int] = {}
        
        for cell in result.cells:
            for wid in cell.word_ids:
                assigned_ids.add(wid)
                id_counts[wid] = id_counts.get(wid, 0) + 1
        
        # Compute metrics
        coverage = len(assigned_ids) / total_words if total_words > 0 else 1.0
        
        # Duplication: words used more than once
        duplicated = sum(1 for wid, count in id_counts.items() if count > 1)
        duplication_ratio = duplicated / len(assigned_ids) if assigned_ids else 0.0
        
        # Row sanity: consistent column count
        row_sanity_score = self._compute_row_sanity(result)
        
        # Empty cell ratio
        total_cells = len(result.cells)
        empty_cells = sum(1 for c in result.cells if not c.text.strip())
        empty_cell_ratio = empty_cells / total_cells if total_cells > 0 else 0.0
        
        # Unassigned words
        all_word_ids = {w.id for w in words_in_bbox}
        unassigned = list(all_word_ids - assigned_ids)
        
        # Determine pass/fail
        failure_reasons = []
        
        if coverage < self.min_coverage:
            failure_reasons.append(f"coverage={coverage:.2f} < {self.min_coverage}")
        
        if duplication_ratio > self.max_duplication:
            failure_reasons.append(f"duplication={duplication_ratio:.2f} > {self.max_duplication}")
        
        if self.require_sanity and row_sanity_score < 0.8:
            failure_reasons.append(f"row_sanity={row_sanity_score:.2f} < 0.8")
        
        passed = len(failure_reasons) == 0
        
        return TableQAMetrics(
            coverage=coverage,
            duplication_ratio=duplication_ratio,
            row_sanity_score=row_sanity_score,
            empty_cell_ratio=empty_cell_ratio,
            passed=passed,
            failure_reasons=failure_reasons,
            unassigned_word_ids=unassigned,
        )
    
    def _get_words_in_bbox(
        self,
        primitives: TablePrimitives,
        bbox: BBoxPDF,
    ) -> List:
        """Get words whose center is inside the bbox."""
        x0, y0, x1, y1 = bbox
        words_in = []
        
        for word in primitives.words:
            # Check if word center is in bbox
            wx0, wy0, wx1, wy1 = word.bbox
            cx = (wx0 + wx1) / 2
            cy = (wy0 + wy1) / 2
            
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                words_in.append(word)
        
        return words_in
    
    def _compute_row_sanity(self, result: TableResult) -> float:
        """
        Compute row sanity score (how consistent is column count).
        
        Returns 1.0 if all rows have same column count, lower otherwise.
        """
        if not result.cells:
            return 1.0
        
        # Group cells by row
        rows: Dict[int, int] = {}
        for cell in result.cells:
            rows[cell.row] = rows.get(cell.row, 0) + 1
        
        if not rows:
            return 1.0
        
        col_counts = list(rows.values())
        if len(col_counts) == 1:
            return 1.0
        
        # Mode column count
        from collections import Counter
        counter = Counter(col_counts)
        mode_count = counter.most_common(1)[0][1]
        
        # Sanity = fraction of rows matching mode
        return mode_count / len(col_counts)
    
    def suggest_action(
        self,
        qa: TableQAMetrics,
        method: str,
    ) -> Dict:
        """
        Suggest action based on QA failure.
        
        Args:
            qa: QA metrics
            method: Current extraction method
        
        Returns:
            Action dict with strategy suggestion
        """
        if qa.passed:
            return {"action": "accept", "reason": "QA passed"}
        
        # Low coverage suggests wrong strategy
        if qa.coverage < 0.5:
            if "kv" not in method:
                return {"action": "rerun", "strategy": "kv", "reason": "Low coverage, try KV"}
            else:
                return {"action": "rerun", "strategy": "tsr", "reason": "KV failed, try TSR"}
        
        # High duplication suggests overlapping cells
        if qa.duplication_ratio > 0.1:
            return {"action": "rerun", "strategy": "ruled", "reason": "High duplication, try ruled grid"}
        
        # Row sanity issue
        if qa.row_sanity_score < 0.8:
            return {"action": "rerun", "strategy": "tsr", "reason": "Row sanity issue, try TSR"}
        
        # Default: escalate to TSR
        return {"action": "escalate", "strategy": "tsr", "reason": "Unknown issue"}
