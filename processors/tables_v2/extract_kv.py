"""
KV Table Extractor - Extract key-value / 2-column tables using native text.

This is the "killer feature" for datasheet-style PDFs:
1. Group words into rows by Y-baseline
2. Split into 2 columns using max X-gap
3. Handle multi-line values and section headers
"""

import logging
from typing import List, Tuple, Optional
import uuid

from .types import (
    BBoxPDF, 
    TablePrimitives, 
    TableCell, 
    TableResult, 
    TableType,
    TableQAMetrics,
    WordSpan,
)

logger = logging.getLogger(__name__)


class TableExtractorKV:
    """
    Extract 2-column key-value tables from native PDF text.
    
    Works best for:
    - Datasheets with "Property: Value" format
    - Spec tables with left-aligned keys
    - Forms with label/field pairs
    """
    
    def __init__(
        self,
        header_width_ratio: float = 0.7,  # Row spans >70% width is header
        multiline_join: bool = True,  # Join multi-line values
        row_tolerance_factor: float = 0.6,  # Tolerance = factor * median_font_size
    ):
        self.header_width_ratio = header_width_ratio
        self.multiline_join = multiline_join
        self.row_tolerance_factor = row_tolerance_factor
    
    def extract(
        self,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        table_id: Optional[str] = None,
    ) -> TableResult:
        """
        Extract KV table from the given region.
        
        Args:
            bbox: Table bounding box in PDF coords
            primitives: Page primitives
            table_id: Optional unique ID for the table
        
        Returns:
            TableResult with 2-column cells
        """
        import time
        start_time = time.time()
        
        table_id = table_id or str(uuid.uuid4())[:8]
        
        # Get words in bbox
        words = primitives.get_words_in_bbox(bbox, overlap_threshold=0.5)
        
        if not words:
            return self._empty_result(table_id, bbox, start_time)
        
        bbox_width = bbox[2] - bbox[0]
        
        # Step 1: Group words into rows
        row_tolerance = self._estimate_row_tolerance(words)
        rows = self._group_into_rows(words, row_tolerance)
        
        if not rows:
            return self._empty_result(table_id, bbox, start_time)
        
        # Step 2: Find the column split point
        split_x = self._find_column_split(words, bbox)
        
        if split_x is None:
            # Can't find good split, treat as single-column
            return self._extract_single_column(rows, table_id, bbox, start_time)
        
        # Step 3: Build KV cells
        cells = []
        row_idx = 0
        used_word_ids = set()
        
        for row_words in rows:
            row_bbox = self._row_bbox(row_words)
            row_width = row_bbox[2] - row_bbox[0]
            
            # Check if this is a section header (spans most of width)
            if row_width / bbox_width > self.header_width_ratio:
                # Emit as header row
                text = " ".join(w.text for w in sorted(row_words, key=lambda w: w.bbox[0]))
                word_ids = [w.id for w in row_words]
                used_word_ids.update(word_ids)
                
                cells.append(TableCell(
                    row=row_idx,
                    col=0,
                    colspan=2,
                    bbox_pdf=row_bbox,
                    text=text,
                    word_ids=word_ids,
                    is_header=True,
                ))
                row_idx += 1
                continue
            
            # Split into left (key) and right (value) columns
            left_words = [w for w in row_words if (w.bbox[0] + w.bbox[2]) / 2 < split_x]
            right_words = [w for w in row_words if (w.bbox[0] + w.bbox[2]) / 2 >= split_x]
            
            # Multi-line value handling: if only right words and previous row exists
            if self.multiline_join and not left_words and right_words and cells:
                # Join with previous row's value
                prev_cell = cells[-1]
                if prev_cell.col == 1:  # Previous was a value cell
                    value_text = " ".join(w.text for w in sorted(right_words, key=lambda w: w.bbox[0]))
                    prev_cell.text += " " + value_text
                    prev_cell.word_ids.extend(w.id for w in right_words)
                    used_word_ids.update(w.id for w in right_words)
                    continue
            
            # Emit key cell
            if left_words:
                key_text = " ".join(w.text for w in sorted(left_words, key=lambda w: w.bbox[0]))
                key_ids = [w.id for w in left_words]
                used_word_ids.update(key_ids)
                
                cells.append(TableCell(
                    row=row_idx,
                    col=0,
                    bbox_pdf=self._row_bbox(left_words),
                    text=key_text,
                    word_ids=key_ids,
                ))
            
            # Emit value cell
            if right_words:
                value_text = " ".join(w.text for w in sorted(right_words, key=lambda w: w.bbox[0]))
                value_ids = [w.id for w in right_words]
                used_word_ids.update(value_ids)
                
                cells.append(TableCell(
                    row=row_idx,
                    col=1,
                    bbox_pdf=self._row_bbox(right_words),
                    text=value_text,
                    word_ids=value_ids,
                ))
            
            row_idx += 1
        
        # Compute QA metrics
        qa = self._compute_qa(words, cells, used_word_ids)
        
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.KV,
            method="kv_native",
            cells=cells,
            qa=qa,
            num_rows=row_idx,
            num_cols=2,
            router_scores={},
            extraction_time_ms=elapsed,
        )
    
    def _estimate_row_tolerance(self, words: List[WordSpan]) -> float:
        """Estimate row grouping tolerance based on font sizes."""
        font_sizes = [w.font_size for w in words if w.font_size > 0]
        
        if font_sizes:
            median_size = sorted(font_sizes)[len(font_sizes) // 2]
            return max(3.0, median_size * self.row_tolerance_factor)
        
        # Fallback: use y-coordinate differences
        y_baselines = sorted(set(w.bbox[3] for w in words))  # Bottom of each word
        if len(y_baselines) > 1:
            diffs = [y_baselines[i+1] - y_baselines[i] for i in range(len(y_baselines)-1)]
            if diffs:
                return max(3.0, min(diffs) * 0.5)
        
        return 8.0  # Default
    
    def _group_into_rows(
        self, 
        words: List[WordSpan], 
        tolerance: float
    ) -> List[List[WordSpan]]:
        """Group words into rows by Y-baseline."""
        if not words:
            return []
        
        # Sort by Y (top of word)
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
    
    def _find_column_split(
        self, 
        words: List[WordSpan], 
        bbox: BBoxPDF
    ) -> Optional[float]:
        """
        Find the X-coordinate that best splits the table into 2 columns.
        
        Uses the "max gap" method: find the largest horizontal gap between words.
        """
        if len(words) < 2:
            return None
        
        bbox_width = bbox[2] - bbox[0]
        
        # Get x-centers of words
        x_centers = sorted((w.bbox[0] + w.bbox[2]) / 2 for w in words)
        
        if len(x_centers) < 2:
            return None
        
        # Find gaps
        best_gap = 0
        best_split = None
        
        for i in range(len(x_centers) - 1):
            gap = x_centers[i + 1] - x_centers[i]
            mid = (x_centers[i] + x_centers[i + 1]) / 2
            
            # The split should be roughly in the middle third
            relative_pos = (mid - bbox[0]) / bbox_width
            if 0.2 < relative_pos < 0.8 and gap > best_gap:
                best_gap = gap
                best_split = mid
        
        # Require a meaningful gap (at least 5% of bbox width)
        if best_gap < bbox_width * 0.05:
            return None
        
        return best_split
    
    def _row_bbox(self, words: List[WordSpan]) -> BBoxPDF:
        """Compute bounding box for a list of words."""
        if not words:
            return (0, 0, 0, 0)
        
        return (
            min(w.bbox[0] for w in words),
            min(w.bbox[1] for w in words),
            max(w.bbox[2] for w in words),
            max(w.bbox[3] for w in words),
        )
    
    def _extract_single_column(
        self,
        rows: List[List[WordSpan]],
        table_id: str,
        bbox: BBoxPDF,
        start_time: float,
    ) -> TableResult:
        """Fallback: extract as single-column table."""
        cells = []
        used_word_ids = set()
        
        for row_idx, row_words in enumerate(rows):
            text = " ".join(w.text for w in sorted(row_words, key=lambda w: w.bbox[0]))
            word_ids = [w.id for w in row_words]
            used_word_ids.update(word_ids)
            
            cells.append(TableCell(
                row=row_idx,
                col=0,
                bbox_pdf=self._row_bbox(row_words),
                text=text,
                word_ids=word_ids,
            ))
        
        all_words = [w for row in rows for w in row]
        qa = self._compute_qa(all_words, cells, used_word_ids)
        
        import time
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.KV,
            method="kv_single_col",
            cells=cells,
            qa=qa,
            num_rows=len(rows),
            num_cols=1,
            extraction_time_ms=elapsed,
        )
    
    def _compute_qa(
        self,
        all_words: List[WordSpan],
        cells: List[TableCell],
        used_word_ids: set,
    ) -> TableQAMetrics:
        """Compute QA metrics for the extraction."""
        total_words = len(all_words)
        assigned_words = len(used_word_ids)
        
        # Check for duplicates (word in multiple cells)
        word_to_cells = {}
        for cell in cells:
            for wid in cell.word_ids:
                word_to_cells.setdefault(wid, []).append(cell)
        
        duplicated = sum(1 for wid, cell_list in word_to_cells.items() if len(cell_list) > 1)
        
        coverage = assigned_words / total_words if total_words > 0 else 0.0
        dup_ratio = duplicated / assigned_words if assigned_words > 0 else 0.0
        unassigned = [w.id for w in all_words if w.id not in used_word_ids]
        
        # Sanity checks
        failure_reasons = []
        if coverage < 0.9:
            failure_reasons.append(f"Low coverage: {coverage:.2f}")
        if dup_ratio > 0.02:
            failure_reasons.append(f"High duplication: {dup_ratio:.2f}")
        
        return TableQAMetrics(
            coverage=coverage,
            duplication_ratio=dup_ratio,
            row_sanity_score=1.0,  # KV has consistent 2-column structure
            empty_cell_ratio=0.0,
            unassigned_word_ids=unassigned,
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
    
    def _empty_result(
        self, 
        table_id: str, 
        bbox: BBoxPDF, 
        start_time: float
    ) -> TableResult:
        """Return empty result when no words found."""
        import time
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.KV,
            method="kv_empty",
            cells=[],
            qa=TableQAMetrics(passed=False, failure_reasons=["No words found"]),
            extraction_time_ms=elapsed,
        )
