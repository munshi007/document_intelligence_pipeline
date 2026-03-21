"""
Tables v2 - Advanced Table Extraction System

This package provides a deterministic, native-first table extraction pipeline:
1. PdfPrimitivesExtractor: Extract words/drawings from PDF
2. TableBboxRefiner: Snap bboxes to word unions
3. TableTypeRouter: Classify tables as RULED/KV/COMPLEX
4. Extractors: KV (clustering), Ruled (vector grid), TSR (model fallback)
5. TableQA: Coverage and quality metrics
6. VLM Planner/Referee: Optional LLM guidance

Usage:
    from processors.tables_v2 import TableCoordinator
    import fitz
    
    doc = fitz.open("document.pdf")
    page = doc[0]
    
    coordinator = TableCoordinator()
    result = coordinator.extract_table(page, (x0, y0, x1, y1))
    print(result.to_markdown())
"""

# Type exports
from .types import (
    BBoxPDF,
    WordSpan,
    DrawingPrimitive,
    TablePrimitives,
    TableType,
    TableCell,
    TableResult,
    TableQAMetrics,
    PageInfo,
)

# Component exports
from .primitives import PdfPrimitivesExtractor
from .refiner import TableBboxRefiner
from .router import TableTypeRouter
from .extract_kv import TableExtractorKV
from .extract_ruled import TableExtractorRuled
from .extract_tsr import TableExtractorTSR
from .qa import TableQA
from .coordinator import TableCoordinator

# TSR exports
from .tsr import TSREngine, CellPx

__all__ = [
    # Types
    "BBoxPDF",
    "WordSpan",
    "DrawingPrimitive",
    "TablePrimitives",
    "TableType",
    "TableCell",
    "TableResult",
    "TableQAMetrics",
    "PageInfo",
    # Components
    "PdfPrimitivesExtractor",
    "TableBboxRefiner",
    "TableTypeRouter",
    "TableExtractorKV",
    "TableExtractorRuled",
    "TableExtractorTSR",
    "TableQA",
    "TableCoordinator",
    # TSR
    "TSREngine",
    "CellPx",
]
