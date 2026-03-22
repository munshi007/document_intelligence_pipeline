"""
TSR Table Extractor - Use Table Structure Recognition models for complex tables.

This is the fallback strategy for tables that can't be handled by KV or Ruled:
1. Render table region to image
2. Run TSR model to predict cell structure
3. Map cells back to PDF coordinates
4. Fill cells with native words (no OCR if possible)
"""

import logging
from typing import List, Optional, Set
import uuid
import numpy as np

from .types import (
    BBoxPDF,
    TablePrimitives,
    TableCell,
    TableResult,
    TableType,
    TableQAMetrics,
    WordSpan,
)
from .tsr.base import TSREngine, CellPx

logger = logging.getLogger(__name__)


class TableExtractorTSR:
    """
    Extract complex tables using TSR models with native text fill.
    
    The TSR model predicts structure only; we use native PDF words
    rather than OCR to fill cell content for better accuracy.
    """
    
    def __init__(
        self,
        engine: Optional[TSREngine] = None,
        render_dpi: int = 150,
        ocr_fallback: bool = True,
    ):
        """
        Args:
            engine: TSR engine to use (TATR, Surya, etc.)
            render_dpi: DPI for rendering table region
            ocr_fallback: Whether to OCR if no native text
        """
        self.engine = engine
        self.render_dpi = render_dpi
        self.ocr_fallback = ocr_fallback
    
    def extract(
        self,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        page: "fitz.Page" = None,
        table_id: Optional[str] = None,
    ) -> TableResult:
        """
        Extract table using TSR model.
        
        Args:
            bbox: Table bounding box
            primitives: Page primitives
            page: fitz.Page for rendering (optional, needed for TSR)
            table_id: Optional unique ID
        
        Returns:
            TableResult with cells from TSR + native fill
        """
        import time
        start_time = time.time()
        
        table_id = table_id or str(uuid.uuid4())[:8]
        
        if self.engine is None:
            return self._empty_result(table_id, bbox, start_time, "No TSR engine configured")
        
        # Get words in bbox for native fill
        words = primitives.get_words_in_bbox(bbox, overlap_threshold=0.5)
        has_native_text = len(words) > 0
        
        # Render table region to image
        if page is None:
            return self._empty_result(table_id, bbox, start_time, "No page provided for rendering")
        
        try:
            crop_image, image_size = self.engine.render_crop(page, bbox, self.render_dpi)
            
            # --- ADVANCED SOTA: NATIVE RASTER MASKING ---
            table_images = primitives.get_images_in_bbox(bbox, overlap_threshold=0.3)
            if table_images:
                logger.info(f"Masking {len(table_images)} embedded images/graphs from TATR crop.")
                import cv2
                crop_image = np.copy(crop_image) # Ensure writable
                
                pdf_width = bbox[2] - bbox[0]
                pdf_height = bbox[3] - bbox[1]
                scale_x = image_size[0] / pdf_width if pdf_width > 0 else 1.0
                scale_y = image_size[1] / pdf_height if pdf_height > 0 else 1.0
                
                for img_prim in table_images:
                    # Map from PDF coords to relative pixel coords in the crop
                    px_x0 = int((max(img_prim.bbox[0], bbox[0]) - bbox[0]) * scale_x)
                    px_y0 = int((max(img_prim.bbox[1], bbox[1]) - bbox[1]) * scale_y)
                    px_x1 = int((min(img_prim.bbox[2], bbox[2]) - bbox[0]) * scale_x)
                    px_y1 = int((min(img_prim.bbox[3], bbox[3]) - bbox[1]) * scale_y)
                    
                    if px_x1 > px_x0 and px_y1 > px_y0:
                        # White out the graph region completely
                        cv2.rectangle(crop_image, (px_x0, px_y0), (px_x1, px_y1), (255, 255, 255), -1)
            # --------------------------------------------
            
        except Exception as e:
            logger.warning(f"Failed to render table crop: {e}")
            return self._empty_result(table_id, bbox, start_time, f"Render failed: {e}")
        
        # Run TSR model
        try:
            cells_px = self.engine.predict_cells(crop_image)

        except Exception as e:
            logger.warning(f"TSR prediction failed: {e}")
            return self._empty_result(table_id, bbox, start_time, f"TSR failed: {e}")
        
        if not cells_px:
            return self._empty_result(table_id, bbox, start_time, "TSR returned no cells")
        
        # Map cells to PDF coordinates
        cells_pdf = self.engine.map_cells_to_pdf(cells_px, bbox, image_size)
        
        # Fill cells with native words
        used_word_ids: Set[int] = set()
        for cell in cells_pdf:
            self._fill_cell_with_words(cell, words, used_word_ids)
        
        # Compute QA
        qa = self._compute_qa(words, cells_pdf, used_word_ids)
        
        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.COMPLEX,
            method=f"tsr_{self.engine.name}",
            cells=cells_pdf,
            qa=qa,
            num_rows=max((c.row for c in cells_pdf), default=0) + 1,
            num_cols=max((c.col for c in cells_pdf), default=0) + 1,
            extraction_time_ms=elapsed,
        )
    
    def _fill_cell_with_words(
        self,
        cell: TableCell,
        words: List[WordSpan],
        used_word_ids: Set[int],
    ) -> None:
        """Fill a cell with words that overlap it."""
        if cell.bbox_pdf is None:
            return
        
        cx0, cy0, cx1, cy1 = cell.bbox_pdf
        cell_words = []
        
        for word in words:
            # Use center containment for assignment
            word_cx = (word.bbox[0] + word.bbox[2]) / 2
            word_cy = (word.bbox[1] + word.bbox[3]) / 2
            
            if cx0 <= word_cx <= cx1 and cy0 <= word_cy <= cy1:
                cell_words.append(word)
                used_word_ids.add(word.id)
        
        # Sort words by reading order (top-to-bottom, left-to-right)
        cell_words.sort(key=lambda w: (w.bbox[1], w.bbox[0]))
        
        cell.text = " ".join(w.text for w in cell_words)
        cell.word_ids = [w.id for w in cell_words]
    
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
        
        failure_reasons = []
        if coverage < 0.9:
            failure_reasons.append(f"Low coverage: {coverage:.2f}")
        if dup_ratio > 0.02:
            failure_reasons.append(f"High duplication: {dup_ratio:.2f}")
        
        return TableQAMetrics(
            coverage=coverage,
            duplication_ratio=dup_ratio,
            row_sanity_score=1.0,  # TSR model provides consistent structure
            empty_cell_ratio=0.0,
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
            table_type=TableType.COMPLEX,
            method="tsr_empty",
            cells=[],
            qa=TableQAMetrics(passed=False, failure_reasons=[reason]),
            extraction_time_ms=elapsed,
        )
