"""
Ruled Table Extractor - Extract grid tables using vector lines from PDF.

This extractor works best for tables with visible borders:
1. Extract horizontal/vertical vector lines from PDF drawings
2. Merge collinear segments
3. Build grid from unique X/Y boundaries
4. Fill cells with overlapping native words
"""

import logging
from typing import List, Tuple, Optional, Set
import uuid

from .types import (
    BBoxPDF,
    TablePrimitives,
    TableCell,
    TableResult,
    TableType,
    TableQAMetrics,
    WordSpan,
    DrawingPrimitive,
)

logger = logging.getLogger(__name__)


class TableExtractorRuled:
    """
    Extract ruled/grid tables using PDF vector line primitives.
    
    Works best for:
    - Tables with visible borders
    - Grid tables with clear row/column lines
    - Forms with box structure
    """
    
    def __init__(
        self,
        min_line_length: float = 10.0,  # Minimum line length (points)
        merge_tolerance: float = 3.0,  # Tolerance for merging collinear lines
        boundary_tolerance: float = 5.0,  # Tolerance for unique boundary detection
    ):
        self.min_line_length = min_line_length
        self.merge_tolerance = merge_tolerance
        self.boundary_tolerance = boundary_tolerance
    
    def extract(
        self,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        table_id: Optional[str] = None,
    ) -> TableResult:
        """
        Extract ruled table from the given region.
        
        Args:
            bbox: Table bounding box in PDF coords
            primitives: Page primitives
            table_id: Optional unique ID
        
        Returns:
            TableResult with grid cells
        """
        import time
        start_time = time.time()
        
        table_id = table_id or str(uuid.uuid4())[:8]
        
        # Get drawings and words in bbox
        drawings = primitives.get_drawings_in_bbox(bbox, overlap_threshold=0.3)
        words = primitives.get_words_in_bbox(bbox, overlap_threshold=0.5)
        
        if not drawings:
            return self._empty_result(table_id, bbox, start_time, "No drawings found")
        
        # Step 1: Extract and filter lines
        h_lines, v_lines = self._extract_lines(drawings, bbox)
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            return self._empty_result(table_id, bbox, start_time, "Insufficient grid lines")
        
        # Step 2: Merge collinear segments
        h_lines = self._merge_collinear(h_lines, is_horizontal=True)
        v_lines = self._merge_collinear(v_lines, is_horizontal=False)
        
        # Step 3: Find unique boundaries
        y_boundaries = self._find_boundaries([l[1] for l in h_lines], self.boundary_tolerance)
        x_boundaries = self._find_boundaries([l[0] for l in v_lines], self.boundary_tolerance)
        
        if len(y_boundaries) < 2 or len(x_boundaries) < 2:
            return self._empty_result(table_id, bbox, start_time, "Insufficient boundaries")
        
        # Step 4: Build grid cells
        cells, used_word_ids = self._build_grid_cells(
            x_boundaries, y_boundaries, words
        )
        
        # Compute QA
        qa = self._compute_qa(words, cells, used_word_ids)
        
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.RULED,
            method="ruled_vector",
            cells=cells,
            qa=qa,
            num_rows=len(y_boundaries) - 1,
            num_cols=len(x_boundaries) - 1,
            extraction_time_ms=elapsed,
        )
    
    def _extract_lines(
        self,
        drawings: List[DrawingPrimitive],
        bbox: BBoxPDF,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        Extract horizontal and vertical lines from drawings.
        
        Returns:
            Tuple of (h_lines, v_lines) where each line is (x, y, length) for h
            or (x, y, length) for v
        """
        h_lines = []  # (y, x_start, x_end)
        v_lines = []  # (x, y_start, y_end)
        
        for d in drawings:
            if d.kind != "line" or d.length < self.min_line_length:
                continue
            
            if len(d.points) < 2:
                continue
            
            p1, p2 = d.points[0], d.points[1]
            
            if d.is_horizontal:
                y = (p1[1] + p2[1]) / 2
                x_start = min(p1[0], p2[0])
                x_end = max(p1[0], p2[0])
                h_lines.append((y, x_start, x_end))
            
            elif d.is_vertical:
                x = (p1[0] + p2[0]) / 2
                y_start = min(p1[1], p2[1])
                y_end = max(p1[1], p2[1])
                v_lines.append((x, y_start, y_end))
        
        # Also check for rectangles (they contribute 4 lines each)
        for d in drawings:
            if d.kind == "rect":
                x0, y0, x1, y1 = d.bbox
                # Add 4 lines
                h_lines.append((y0, x0, x1))  # Top
                h_lines.append((y1, x0, x1))  # Bottom
                v_lines.append((x0, y0, y1))  # Left
                v_lines.append((x1, y0, y1))  # Right
        
        return h_lines, v_lines
    
    def _merge_collinear(
        self,
        lines: List[Tuple],
        is_horizontal: bool,
    ) -> List[Tuple]:
        """Merge collinear line segments that are close together."""
        if not lines:
            return []
        
        # Group by primary coordinate (y for horizontal, x for vertical)
        groups = {}
        for line in lines:
            key = round(line[0] / self.merge_tolerance) * self.merge_tolerance
            groups.setdefault(key, []).append(line)
        
        merged = []
        for key, group in groups.items():
            # Merge segments in each group
            if is_horizontal:
                # Lines are (y, x_start, x_end)
                avg_y = sum(l[0] for l in group) / len(group)
                x_start = min(l[1] for l in group)
                x_end = max(l[2] for l in group)
                merged.append((avg_y, x_start, x_end))
            else:
                # Lines are (x, y_start, y_end)
                avg_x = sum(l[0] for l in group) / len(group)
                y_start = min(l[1] for l in group)
                y_end = max(l[2] for l in group)
                merged.append((avg_x, y_start, y_end))
        
        return sorted(merged, key=lambda l: l[0])
    
    def _find_boundaries(
        self,
        positions: List[float],
        tolerance: float,
    ) -> List[float]:
        """Find unique boundary positions, merging close values."""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        boundaries = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - boundaries[-1] > tolerance:
                boundaries.append(pos)
            else:
                # Merge by averaging
                boundaries[-1] = (boundaries[-1] + pos) / 2
        
        return boundaries
    
    def _build_grid_cells(
        self,
        x_boundaries: List[float],
        y_boundaries: List[float],
        words: List[WordSpan],
    ) -> Tuple[List[TableCell], Set[int]]:
        """Build cells from grid boundaries and fill with words."""
        cells = []
        used_word_ids = set()
        
        for row_idx in range(len(y_boundaries) - 1):
            y0 = y_boundaries[row_idx]
            y1 = y_boundaries[row_idx + 1]
            
            for col_idx in range(len(x_boundaries) - 1):
                x0 = x_boundaries[col_idx]
                x1 = x_boundaries[col_idx + 1]
                
                cell_bbox = (x0, y0, x1, y1)
                
                # Find words that belong to this cell
                cell_words = self._get_words_for_cell(words, cell_bbox)
                
                # Sort words left-to-right, top-to-bottom
                cell_words.sort(key=lambda w: (w.bbox[1], w.bbox[0]))
                
                text = " ".join(w.text for w in cell_words)
                word_ids = [w.id for w in cell_words]
                used_word_ids.update(word_ids)
                
                cells.append(TableCell(
                    row=row_idx,
                    col=col_idx,
                    bbox_pdf=cell_bbox,
                    text=text,
                    word_ids=word_ids,
                    is_header=(row_idx == 0),  # Assume first row is header
                ))
        
        return cells, used_word_ids
    
    def _get_words_for_cell(
        self,
        words: List[WordSpan],
        cell_bbox: BBoxPDF,
    ) -> List[WordSpan]:
        """Get words that belong to a cell (based on center containment)."""
        result = []
        cx0, cy0, cx1, cy1 = cell_bbox
        
        for word in words:
            # Use word center for assignment
            word_cx = (word.bbox[0] + word.bbox[2]) / 2
            word_cy = (word.bbox[1] + word.bbox[3]) / 2
            
            if cx0 <= word_cx <= cx1 and cy0 <= word_cy <= cy1:
                result.append(word)
        
        return result
    
    def _compute_qa(
        self,
        all_words: List[WordSpan],
        cells: List[TableCell],
        used_word_ids: Set[int],
    ) -> TableQAMetrics:
        """Compute QA metrics."""
        total_words = len(all_words)
        assigned_words = len(used_word_ids)
        
        # Check duplicates
        word_to_cells = {}
        for cell in cells:
            for wid in cell.word_ids:
                word_to_cells.setdefault(wid, []).append(cell)
        
        duplicated = sum(1 for wid, cell_list in word_to_cells.items() if len(cell_list) > 1)
        
        coverage = assigned_words / total_words if total_words > 0 else 0.0
        dup_ratio = duplicated / assigned_words if assigned_words > 0 else 0.0
        unassigned = [w.id for w in all_words if w.id not in used_word_ids]
        
        # Count empty cells
        empty_cells = sum(1 for c in cells if not c.text.strip())
        empty_ratio = empty_cells / len(cells) if cells else 0.0
        
        failure_reasons = []
        if coverage < 0.9:
            failure_reasons.append(f"Low coverage: {coverage:.2f}")
        if dup_ratio > 0.02:
            failure_reasons.append(f"High duplication: {dup_ratio:.2f}")
        if empty_ratio > 0.5:
            failure_reasons.append(f"Many empty cells: {empty_ratio:.2f}")
        
        return TableQAMetrics(
            coverage=coverage,
            duplication_ratio=dup_ratio,
            row_sanity_score=1.0,  # Grid has uniform structure
            empty_cell_ratio=empty_ratio,
            unassigned_word_ids=unassigned,
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
    
    def _empty_result(
        self,
        table_id: str,
        bbox: BBoxPDF,
        start_time: float,
        reason: str,
    ) -> TableResult:
        """Return empty result."""
        import time
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.RULED,
            method="ruled_empty",
            cells=[],
            qa=TableQAMetrics(passed=False, failure_reasons=[reason]),
            extraction_time_ms=elapsed,
        )
