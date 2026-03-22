"""
Table Extractor - Native-First Production Wrapper
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import TABLE_CONFIG
from common.utils import save_image

# Tables v2 - Native-first table extraction
try:
    from processors.tables_v2 import TableCoordinator
except ImportError:
    TableCoordinator = None

logger = logging.getLogger(__name__)

class TableExtractor:
    """
    Production-ready wrapper for Tables v2 extraction.
    Ensures all table regions are routed through the native-first coordinator.
    """

    def __init__(self, output_paths: Dict[str, Any], structure_model: Optional[Any] = None, vlm_client: Optional[Any] = None, strategy: Optional[str] = None):
        """Initialize TableExtractor with shared components."""
        self.output_paths = output_paths
        self.structure_model = structure_model
        self.vlm_client = vlm_client
        self.strategy = strategy
        
        # Initialize Tables v2 coordinator
        self.tables_v2_coordinator = None
        if TableCoordinator:
            try:
                # Use TATR as TSR backend if provided
                tsr_engine = None
                if structure_model:
                    from processors.tables_v2.tsr.tatr import TATREngine
                    model, processor = structure_model.get_components()
                    if model and processor:
                        tsr_engine = TATREngine(model, processor)
                
                self.tables_v2_coordinator = TableCoordinator(
                    tsr_engine=tsr_engine,
                    vlm_client=vlm_client
                )
                logger.info("TableExtractor: Tables v2 coordinator initialized.")
            except Exception as e:
                logger.error(f"TableExtractor: Tables v2 initialization failed: {e}")

    def extract_table_structure(
        self,
        page_image: np.ndarray,
        table_bbox: List[float],
        page_num: int,
        table_num: int,
        pdf_path: Optional[str] = None,
        pdf_page_num: Optional[int] = None,
        pdf_bbox: Optional[List[float]] = None,
        fitz_page: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Routes every detected table to the Tables v2 coordinator.
        """
        if self.tables_v2_coordinator is None:
            return {"rows": [], "method": "unavailable", "error": "TableCoordinator not initialized"}

        # 1. Image Crop Preprocessing (Padding & Save)
        TABLE_PADDING = TABLE_CONFIG.get('bbox_padding', 20)
        img_h, img_w = page_image.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in table_bbox]
        
        x1, y1 = max(0, x1 - TABLE_PADDING), max(0, y1 - TABLE_PADDING)
        x2, y2 = min(img_w, x2 + TABLE_PADDING), min(img_h, y2 + TABLE_PADDING)

        table_crop = page_image[y1:y2, x1:x2]
        table_filename = f"table_page_{page_num:02d}_{table_num:02d}.png"
        tables_dir = self.output_paths.get("tables", Path("Output/tables"))
        table_path = tables_dir / table_filename
        save_image(table_crop, table_path, f"table image: {table_filename}")

        # 2. Strict Tables v2 Routing
        if not fitz_page or not pdf_bbox:
            logger.error(f"TableExtractor: Missing native PDF metadata for table {table_num} on page {page_num}")
            return {"rows": [], "method": "error", "error": "Missing fitz_page or pdf_bbox"}

        try:
            logger.info(f"[tables_v2] Extracting: Table {table_num} (Page {page_num})")
            result = self.tables_v2_coordinator.extract_table(
                fitz_page,
                tuple(pdf_bbox),
                table_id=f"t{table_num:02d}"
            )
            
            # Map TableResult to legacy row format for the MarkdownRenderer
            rows = []
            if result.cells:
                grid = {(cell.row, cell.col): cell.text for cell in result.cells}
                max_row = max(r for r, c in grid.keys())
                max_col = max(c for r, c in grid.keys())
                for r in range(max_row + 1):
                    row = [grid.get((r, c), "") for c in range(max_col + 1)]
                    rows.append(row)

            return {
                "rows": rows,
                "method": f"tables_v2_{result.method}",
                "table_type": result.table_type.value,
                "num_rows": result.num_rows,
                "num_cols": result.num_cols,
                "qa_passed": result.qa.passed,
                "table_image_path": str(table_path)
            }
            
        except Exception as e:
            logger.error(f"TableExtractor: Tables v2 extraction failed: {e}")
            return {"rows": [], "method": "tables_v2_error", "error": str(e), "table_image_path": str(table_path)}
