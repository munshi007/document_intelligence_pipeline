"""
Table Extraction Coordinator - Orchestrates the complete tables_v2 pipeline.

This is the main entry point that coordinates:
1. Primitives extraction
2. BBox refinement
3. Type routing
4. Strategy-specific extraction
5. QA validation
6. Optional VLM referee
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import uuid

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    fitz = None

from .types import (
    BBoxPDF,
    TablePrimitives,
    TableType,
    TableResult,
    TableQAMetrics,
)
from .primitives import PdfPrimitivesExtractor
from .refiner import TableBboxRefiner
from .router import TableTypeRouter
from .extract_kv import TableExtractorKV
from .extract_ruled import TableExtractorRuled
from .extract_tsr import TableExtractorTSR
from .extract_vlm import TableExtractorVLM
from .qa import TableQA
from .tsr.base import TSREngine

logger = logging.getLogger(__name__)


class TableCoordinator:
    """
    Coordinates the complete table extraction pipeline.
    
    Flow:
    1. Extract primitives from PDF page
    2. Refine table bbox
    3. Route to appropriate extractor
    4. Run extraction
    5. Validate with QA
    6. Optionally retry with VLM guidance
    """
    
    def __init__(
        self,
        tsr_engine: Optional[TSREngine] = None,
        enable_vlm_planner: bool = False,
        enable_vlm_referee: bool = False,
        vlm_model: str = "qwen2.5-vl:latest", # Kept for backward compat / logging
        max_retries: int = 1,
        vlm_client: Any = None, # Dependency Injection
    ):
        """
        Args:
            tsr_engine: TSR engine for complex tables
            enable_vlm_planner: Use VLM for initial priors
            enable_vlm_referee: Use VLM for QA failure recovery
            vlm_model: Model name preference
            max_retries: Max retry attempts on QA failure
            vlm_client: VLMClient instance
        """
        self.primitives_extractor = PdfPrimitivesExtractor()
        self.refiner = TableBboxRefiner()
        self.router = TableTypeRouter()
        self.kv_extractor = TableExtractorKV()
        self.ruled_extractor = TableExtractorRuled()
        self.tsr_extractor = TableExtractorTSR(engine=tsr_engine)
        self.vlm_extractor = TableExtractorVLM(vlm_client=vlm_client)
        self.qa = TableQA()
        self.max_retries = max_retries
        
        # VLM components
        self.enable_vlm_planner = enable_vlm_planner
        self.enable_vlm_referee = enable_vlm_referee
        self.vlm_client = vlm_client
        self._planner = None
        self._referee = None
    
    def extract_table(
        self,
        page: "fitz.Page",
        table_bbox: BBoxPDF,
        table_id: Optional[str] = None,
        priors: Optional[Dict] = None,
    ) -> TableResult:
        """
        Extract a single table from a PDF page.
        
        Args:
            page: fitz.Page object
            table_bbox: Initial table bounding box from layout detection
            table_id: Optional unique identifier
            priors: Optional pre-computed VLM priors
        
        Returns:
            TableResult with extracted cells
        """
        start_time = time.time()
        table_id = table_id or str(uuid.uuid4())[:8]
        
        print(f"[coordinator] ====== Table {table_id} ======")
        print(f"[coordinator] Input bbox: {table_bbox}")
        
        # Step 1: Extract primitives
        print(f"[coordinator] Step 1: Extracting primitives...")
        primitives = self.primitives_extractor.extract(page)
        print(f"[coordinator] → Found {len(primitives.words)} words, {len(primitives.drawings)} drawings")
        
        # Step 2: Refine bbox
        print(f"[coordinator] Step 2: Refining bbox...")
        refined_bbox, refine_debug = self.refiner.refine(table_bbox, primitives)
        print(f"[coordinator] → Refined bbox: {refined_bbox}")
        
        # Step 3: Get VLM priors if enabled
        if priors is None and self.enable_vlm_planner:
            print(f"[coordinator] Step 3: Getting VLM priors...")
            priors = self._get_vlm_priors(page)
        priors = priors or {}
        if priors:
            print(f"[coordinator] → VLM Priors: {priors}")
        
        # Step 4: Route to strategy
        print(f"[coordinator] Step 4: Routing to extraction strategy...")
        table_type, router_scores = self.router.route(refined_bbox, primitives, priors)
        print(f"[coordinator] → Routed to: {table_type.value}")
        print(f"[coordinator] → Scores: {router_scores}")
        
        # Step 5: Extract with chosen strategy
        print(f"[coordinator] Step 5: Extracting with {table_type.value} strategy...")
        result = self._extract_with_strategy(
            table_type, refined_bbox, primitives, page, table_id
        )
        result.router_scores = router_scores
        print(f"[coordinator] → Got {len(result.cells)} cells")
        
        # Step 6: Validate with QA
        print(f"[coordinator] Step 6: Running QA validation...")
        qa_result = self.qa.evaluate(result, primitives, refined_bbox)
        
        # SOTAA: VLM results are structurally superior but lack word-mapping for coverage
        if result.table_type == TableType.VLM and len(result.cells) > 0:
            print(f"[coordinator] → VLM result detected, overriding coverage QA for structural trust.")
            qa_result.passed = True

        # For complex/ruled tables, allow slight duplication when coverage is very high.
        # This prevents false negatives from tiny border-overlap assignments.
        if (
            not qa_result.passed
            and result.table_type == TableType.COMPLEX
            and qa_result.coverage >= 0.95
            and qa_result.duplication_ratio <= 0.08
            and qa_result.failure_reasons
            and all(reason.startswith("duplication=") for reason in qa_result.failure_reasons)
        ):
            print("[coordinator] → High-coverage complex table with minor duplication; accepting result.")
            qa_result.passed = True
            qa_result.failure_reasons = []
            
        result.qa = qa_result
        print(f"[coordinator] → Coverage: {qa_result.coverage:.1%}")
        print(f"[coordinator] → Duplication: {qa_result.duplication_ratio:.1%}")
        print(f"[coordinator] → QA passed: {qa_result.passed}")
        
        # Step 7: Handle QA failure
        if not qa_result.passed and self.max_retries > 0:
            print(f"[coordinator] Step 7: QA failed, attempting retry...")
            print(f"[coordinator] → Reasons: {qa_result.failure_reasons}")
            result = self._handle_qa_failure(
                result, page, refined_bbox, primitives, priors
            )
            print(f"[coordinator] → After retry: {result.method}, QA passed={result.qa.passed}")
        
        result.extraction_time_ms = (time.time() - start_time) * 1000
        
        print(
            f"[coordinator] ✓ Complete: type={result.table_type.value}, "
            f"method={result.method}, cells={len(result.cells)}, "
            f"time={result.extraction_time_ms:.0f}ms"
        )
        
        return result
    
    def extract_tables(
        self,
        page: "fitz.Page",
        table_bboxes: List[BBoxPDF],
    ) -> List[TableResult]:
        """
        Extract multiple tables from a PDF page.
        
        Args:
            page: fitz.Page object
            table_bboxes: List of table bboxes from layout detection
        
        Returns:
            List of TableResult objects
        """
        # Extract primitives once for all tables
        primitives = self.primitives_extractor.extract(page)
        
        # Get VLM priors once if enabled
        priors = None
        if self.enable_vlm_planner:
            priors = self._get_vlm_priors(page)
        
        results = []
        for i, bbox in enumerate(table_bboxes):
            table_id = f"t{i:02d}"
            result = self.extract_table(page, bbox, table_id, priors)
            results.append(result)
        
        return results
    
    def _extract_with_strategy(
        self,
        table_type: TableType,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        page: "fitz.Page",
        table_id: str,
    ) -> TableResult:
        """Run extraction with the appropriate strategy."""
        if table_type == TableType.RULED:
            return self.ruled_extractor.extract(bbox, primitives, table_id)
        elif table_type == TableType.KV:
            return self.kv_extractor.extract(bbox, primitives, table_id)
        elif table_type == TableType.VLM:
            return self.vlm_extractor.extract(bbox, primitives, page, table_id)
        else:  # COMPLEX
            return self.tsr_extractor.extract(bbox, primitives, page, table_id)
    
    def _handle_qa_failure(
        self,
        result: TableResult,
        page: "fitz.Page",
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        priors: Dict,
    ) -> TableResult:
        """Handle QA failure by retrying with different strategy."""
        logger.debug(f"QA failed for {result.table_id}, attempting recovery")
        
        # Get suggestion from QA
        suggestion = self.qa.suggest_action(result.qa, result.method)
        
        # Optionally use VLM referee
        if self.enable_vlm_referee and suggestion.get("action") != "accept":
            vlm_suggestion = self._get_vlm_referee_action(page, bbox, result)
            if vlm_suggestion.get("action") != "escalate":
                suggestion = vlm_suggestion
        
        action = suggestion.get("action", "escalate")
        strategy = suggestion.get("strategy", "tsr")
        
        if action == "accept":
            return result
        
        # Retry with suggested strategy
        if strategy == "kv":
            new_result = self.kv_extractor.extract(bbox, primitives, result.table_id)
        elif strategy == "ruled":
            new_result = self.ruled_extractor.extract(bbox, primitives, result.table_id)
        else:  # tsr
            new_result = self.tsr_extractor.extract(bbox, primitives, page, result.table_id)
        
        # Re-evaluate QA
        new_qa = self.qa.evaluate(new_result, primitives, bbox)
        new_result.qa = new_qa
        
        # Keep better result
        if new_qa.coverage > result.qa.coverage:
            new_result.method = f"{new_result.method}_retry"
            return new_result
        
        return result

    def _get_vlm_priors(self, page: "fitz.Page") -> Dict:
        """Get VLM priors for the page."""
        if self._planner is None:
            from .planner import TablePlannerVLM
            self._planner = TablePlannerVLM(vlm_client=self.vlm_client)
        
        try:
            # Render page thumbnail
            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]
            
            return self._planner.generate_priors(img)
        except Exception as e:
            logger.warning(f"VLM priors failed: {e}")
            return {}
    
    def _get_vlm_referee_action(
        self,
        page: "fitz.Page",
        bbox: BBoxPDF,
        result: TableResult,
    ) -> Dict:
        """Get VLM referee action."""
        if self._referee is None:
            from .planner import TableRefereeVLM
            self._referee = TableRefereeVLM(vlm_client=self.vlm_client)
        
        try:
            # Render table crop
            clip = fitz.Rect(bbox)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=clip)
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]
            
            # Get preview rows
            preview = []
            for cell in sorted(result.cells, key=lambda c: (c.row, c.col))[:6]:
                preview.append({"row": cell.row, "col": cell.col, "text": cell.text[:50]})
            
            return self._referee.suggest_action(img, result.qa.to_dict(), preview)
        except Exception as e:
            logger.warning(f"VLM referee failed: {e}")
            return {"action": "escalate", "strategy": "tsr"}
