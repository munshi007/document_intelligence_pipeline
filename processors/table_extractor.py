import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import configuration
try:
    from config import TABLE_CONFIG
except ImportError:
    TABLE_CONFIG = {'bbox_padding': 20}

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)

# Tables v2 - Native-first table extraction
TABLES_V2_AVAILABLE = False
try:
    from processors.tables_v2 import TableCoordinator, TableResult
    TABLES_V2_AVAILABLE = True
except ImportError as e:
    pass


from common.utils import save_image


class TableExtractor:
    """
    Native-first table extraction using Tables v2.
    Uses provided TableStructureModel (TATR) as backend for structure recognition.
    """

    def __init__(self, output_paths: Dict[str, Any], structure_model: Optional[Any] = None, vlm_client: Optional[Any] = None, strategy: Optional[str] = None):
        """
        Initialize TableExtractor with shared components.
        
        Args:
            output_paths: Dictionary of output paths.
            structure_model: Initialized TATR model for structure recognition.
            vlm_client: Instance of VLMClient for planner/referee.
            strategy: Optional research strategy (gpt4o, sota_os, fast_os).
        """
        self.output_paths = output_paths
        self.structure_model = structure_model
        self.strategy = strategy
        self.vlm_client = vlm_client
        
        # Cache to track extracted pdfplumber tables (prevent duplicates)
        # Key: (pdf_path, page_num, table_bbox_tuple), Value: extracted rows
        self._extracted_tables_cache = {}
        
        # Initialize Tables v2 coordinator (native-first extraction)
        self.tables_v2_coordinator = None
        self.tables_v2_available = False
        
        if TABLES_V2_AVAILABLE:
            try:
                # Use TATR as TSR backend if available via structure_model
                tsr_engine = None
                if self.structure_model and getattr(self.structure_model, 'available', False):
                    from processors.tables_v2.tsr.tatr import TATREngine
                    model, processor = self.structure_model.get_components()
                    if model and processor:
                        tsr_engine = TATREngine(model, processor)
                        # logger.info("TableExtractor: TATR backend attached from structure_model.")
                
                self.tables_v2_coordinator = TableCoordinator(
                    tsr_engine=tsr_engine,
                    enable_vlm_planner=bool(vlm_client),  # Enable if client provided
                    enable_vlm_referee=bool(vlm_client),
                    vlm_client=vlm_client
                )
                self.tables_v2_available = True
                logger.info("Tables v2 coordinator initialized")
            except Exception as e:
                logger.warning(f"Tables v2 coordinator init failed: {e}")


    # ----------------------------- PUBLIC API ----------------------------- #
    def extract_table_structure(
        self,
        page_image: np.ndarray,
        table_bbox: List[float],
        page_num: int,
        table_num: int,
        doc_profile: Optional[Any] = None,
        pdf_path: Optional[str] = None,
        pdf_page_num: Optional[int] = None,
        pdf_bbox: Optional[List[float]] = None,
        fitz_page: Optional[Any] = None,  # fitz.Page for tables_v2
        vlm_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Smart hybrid table extraction:
        0. Try Tables v2 (native-first extraction) if fitz_page available
        
        Args:
            page_image: Page image as numpy array
            table_bbox: Table bounding box [x1, y1, x2, y2]
            page_num: Page number for output naming
            table_num: Table number on page
            doc_profile: Optional document profile
            pdf_path: Path to original PDF (for pdfplumber)
            pdf_page_num: PDF page number (0-indexed)
            pdf_bbox: PDF-space bbox for native extraction
            fitz_page: Optional fitz.Page for tables_v2 native extraction
        """
        if page_image is None or page_image.size == 0:
            return {"rows": [], "method": "none", "error": "empty page image"}

        # Add padding to table bbox to capture edge text that may be cut off
        # This is especially important for tables without clear borders
        TABLE_PADDING = TABLE_CONFIG.get('bbox_padding', 20)
        
        img_h, img_w = page_image.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in table_bbox]
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - TABLE_PADDING)
        y1 = max(0, y1 - TABLE_PADDING)
        x2 = min(img_w, x2 + TABLE_PADDING)
        y2 = min(img_h, y2 + TABLE_PADDING)

        table_crop = page_image[y1:y2, x1:x2] if page_image is not None else None
        if table_crop is None or table_crop.size == 0:
            return {"rows": [], "method": "none", "error": "empty table crop"}

        # Save table image
        table_filename = f"table_page_{page_num:02d}_{table_num:02d}.png"
        tables_dir = self.output_paths.get("tables", Path("Output/tables"))
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / table_filename
        save_image(table_crop, table_path, f"table image: {table_filename}")

        # TABLES V2 - THE ONLY EXTRACTION METHOD (no fallbacks)
        # All table extraction now goes through the new native-first pipeline
        
        if not self.tables_v2_available:
            print(f"[tables_v2] ERROR: tables_v2 not available - cannot extract tables")
            logger.error("Tables v2 not available - cannot extract tables")
            return {
                "rows": [],
                "method": "unavailable",
                "error": "tables_v2 module not loaded",
                "table_image_path": str(table_path)
            }
        
        if fitz_page is None:
            print(f"[tables_v2] ERROR: fitz_page not provided")
            logger.error("fitz_page not provided - tables_v2 requires fitz.Page object")
            return {
                "rows": [],
                "method": "error",
                "error": "fitz_page required for tables_v2",
                "table_image_path": str(table_path)
            }
        
        if not pdf_bbox:
            # print(f"[tables_v2] ERROR: pdf_bbox not provided")
            # logger.error("pdf_bbox not provided - tables_v2 requires PDF-space coordinates")
            # Allow fallback if pdf_bbox missing? No, user enforced strict V2.
            # But let's log and error gracefully.
             return {
                "rows": [],
                "method": "error", 
                "error": "pdf_bbox required for tables_v2",
                "table_image_path": str(table_path)
            }
        
        # SOTA: GOT-OCR Specialist Table Extraction
        if self.strategy == 'sota_os' and self.vlm_client and 'got-ocr' in getattr(self.vlm_client, 'model', '').lower():
            logger.info(f"[sota_os] Routing table {table_num} to GOT-OCR2.0 for Specialist Extraction...")
            from PIL import Image
            pil_crop = Image.fromarray(table_crop)
            
            # Request formatted markdown table directly from GOT-OCR
            got_result = self.vlm_client.generate_structured(
                image=pil_crop,
                prompt="Extract this table accurately into a Markdown table format.",
                response_model=Any, # We just want the raw table representation 
                is_complex=True,
                metadata=vlm_metadata
            )
            
            if got_result:
                # GOT-OCR often returns the table as a raw string or in a reasoning block
                # For now, we wrap it in a pseudo-row format that MarkdownRenderer can handle
                # Or we can return a specialized 'markdown_table' key
                return {
                    "rows": [], # No legacy rows 
                    "method": "got-ocr-specialist",
                    "markdown_table": got_result,
                    "table_image_path": str(table_path)
                }
        
        # Extract with Tables v2
        print(f"[tables_v2] Starting extraction for table {table_num} on page {page_num}")
        print(f"[tables_v2] PDF bbox: {pdf_bbox}")
        
        table_data = self._extract_with_tables_v2(fitz_page, pdf_bbox, table_num)
        
        if table_data and table_data.get("rows"):
            table_data["table_image_path"] = str(table_path)
            
            # Log extraction details
            print(f"[tables_v2] ✓ Extracted {len(table_data['rows'])} rows, {table_data.get('num_cols', '?')} cols")
            print(f"[tables_v2] Method: {table_data.get('method', 'unknown')}")
            print(f"[tables_v2] Table type: {table_data.get('table_type', 'unknown')}")
            print(f"[tables_v2] Coverage: {table_data.get('coverage', 'N/A')}")
            print(f"[tables_v2] QA passed: {table_data.get('qa_passed', 'unknown')}")
            
            if not table_data.get("qa_passed", True):
                print(f"[tables_v2] QA issues: {table_data.get('qa_reasons', [])}")
            
            return table_data
        
        # Extraction failed
        print(f"[tables_v2] ✗ Extraction failed for table {table_num}")
        return {
            "rows": [],
            "method": "tables_v2_failed",
            "error": table_data.get("error", "No rows extracted") if table_data else "Extraction returned None",
            "table_image_path": str(table_path)
        }

    # --------------------------- INTERNAL METHODS --------------------------- #
    
    def _extract_with_tables_v2(
        self,
        fitz_page: Any,
        pdf_bbox: List[float],
        table_num: int
    ) -> Dict[str, Any]:
        """
        Extract table using Tables v2 coordinator (native-first extraction).
        
        Args:
            fitz_page: fitz.Page object
            pdf_bbox: Table bbox in PDF coordinates
            table_num: Table number for ID
        
        Returns:
            Dict with rows, method, and QA info
        """
        try:
            table_id = f"t{table_num:02d}"
            
            # Convert bbox list to tuple
            bbox_tuple = tuple(pdf_bbox)
            
            # Call coordinator
            result = self.tables_v2_coordinator.extract_table(
                fitz_page,
                bbox_tuple,
                table_id=table_id
            )
            
            # Convert TableResult to legacy row format
            rows = self._convert_table_result_to_rows(result)
            
            return {
                "rows": rows,
                "method": f"tables_v2_{result.method}",
                "table_type": result.table_type.value,
                "coverage": f"{result.qa.coverage:.1%}",
                "qa_passed": result.qa.passed,
                "qa_reasons": result.qa.failure_reasons,
                "num_rows": result.num_rows,
                "num_cols": result.num_cols,
                "extraction_time_ms": result.extraction_time_ms,
            }
            
        except Exception as e:
            logger.warning(f"Tables v2 extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"rows": [], "method": "tables_v2_error", "error": str(e)}
    
    def _convert_table_result_to_rows(self, result: 'TableResult') -> List[List[str]]:
        """
        Convert TableResult cells to legacy row-based format.
        
        Args:
            result: TableResult from tables_v2
        
        Returns:
            List of rows, where each row is a list of cell texts
        """
        if not result.cells:
            return []
        
        # Build grid
        grid = {}
        for cell in result.cells:
            grid[(cell.row, cell.col)] = cell.text
        
        # Determine dimensions
        if not grid:
            return []
            
        max_row = max(r for r, c in grid.keys())
        max_col = max(c for r, c in grid.keys())
        
        # Create dense matrix
        rows = []
        for r in range(max_row + 1):
            row = []
            for c in range(max_col + 1):
                row.append(grid.get((r, c), ""))
            rows.append(row)
            
        return rows
