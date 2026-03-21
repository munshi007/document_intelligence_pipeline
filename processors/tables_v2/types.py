"""
Type definitions for the Tables v2 extraction system.

All geometry is in PDF coordinate space (points, fitz top-left origin).
"""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# Import shared BBox which now supports tuple access and initialization
from common.types import BBox

# Type alias for backward compatibility / readability
BBoxPDF = BBox


class WordSpan(BaseModel):
    """A word extracted from the PDF with its bounding box and metadata."""
    id: int
    text: str
    bbox: BBoxPDF
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0
    font_size: float = 0.0
    font_name: str = ""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DrawingPrimitive(BaseModel):
    """A vector drawing element (line or rectangle) from the PDF."""
    kind: str  # "line" or "rect"
    bbox: BBoxPDF
    points: List[Tuple[float, float]] = Field(default_factory=list)
    width: float = 1.0
    color: Optional[Tuple[float, ...]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def is_horizontal(self) -> bool:
        """Check if this is a roughly horizontal line."""
        if self.kind != "line" or len(self.points) < 2:
            return False
        p1, p2 = self.points[0], self.points[1]
        return abs(p2[1] - p1[1]) < abs(p2[0] - p1[0]) * 0.1
    
    @property
    def is_vertical(self) -> bool:
        """Check if this is a roughly vertical line."""
        if self.kind != "line" or len(self.points) < 2:
            return False
        p1, p2 = self.points[0], self.points[1]
        return abs(p2[0] - p1[0]) < abs(p2[1] - p1[1]) * 0.1
    
    @property
    def length(self) -> float:
        """Get the length of a line primitive."""
        if self.kind != "line" or len(self.points) < 2:
            return 0.0
        p1, p2 = self.points[0], self.points[1]
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PageInfo(BaseModel):
    """Metadata about the PDF page for coordinate transformations."""
    rotation: int = 0
    cropbox: Optional[BBoxPDF] = None
    mediabox: Optional[BBoxPDF] = None
    width: float = 0.0
    height: float = 0.0
    # Transformation matrix for PDF<->fitz coordinate conversion
    transform_matrix: Optional[Tuple[float, ...]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TablePrimitives(BaseModel):
    """All extracted primitives for a page, used for table analysis."""
    words: List[WordSpan] = Field(default_factory=list)
    drawings: List[DrawingPrimitive] = Field(default_factory=list)
    page_info: PageInfo = Field(default_factory=PageInfo)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def get_words_in_bbox(self, bbox: BBoxPDF, overlap_threshold: float = 0.5) -> List[WordSpan]:
        """Get all words that overlap with the given bbox."""
        # Ensure bbox is a BBox object if passed as tuple
        if not isinstance(bbox, BBox):
            bbox = BBox.parse_list_or_tuple(bbox)
            bbox = BBox(**bbox)

        result = []
        for word in self.words:
            if self._compute_overlap_ratio(word.bbox, bbox) >= overlap_threshold:
                result.append(word)
        return result
    
    def get_drawings_in_bbox(self, bbox: BBoxPDF, overlap_threshold: float = 0.3) -> List[DrawingPrimitive]:
        """Get all drawings that overlap with the given bbox."""
         # Ensure bbox is a BBox object
        if not isinstance(bbox, BBox):
            bbox = BBox.parse_list_or_tuple(bbox)
            bbox = BBox(**bbox)

        result = []
        for drawing in self.drawings:
            if self._compute_overlap_ratio(drawing.bbox, bbox) >= overlap_threshold:
                result.append(drawing)
        return result
    
    @staticmethod
    def _compute_overlap_ratio(box1: BBox, box2: BBox) -> float:
        """Compute the overlap ratio of box1 with box2 (how much of box1 is inside box2)."""
        # Box1 and Box2 are BBox objects, supporting index access and .x0 etc
        # Using index access for compatibility if they happen to be tuples (unlikely with Pydantic)
        # But wait, type hint says BBox.
        
        # Safe access using BBox attributes (since we enforce types now)
        x0 = max(box1.x0, box2.x0)
        y0 = max(box1.y0, box2.y0)
        x1 = min(box1.x1, box2.x1)
        y1 = min(box1.y1, box2.y1)
        
        if x0 >= x1 or y0 >= y1:
            return 0.0
        
        intersection = (x1 - x0) * (y1 - y0)
        box1_area = box1.area
        
        if box1_area <= 0:
            return 0.0
        
        return intersection / box1_area


class TableType(str, Enum):
    """Classification of table structure type."""
    RULED = "ruled"      # Grid table with vector lines
    KV = "kv"            # Key-value / 2-column datasheet
    COMPLEX = "complex"  # Complex structure requiring TSR model


class TableCell(BaseModel):
    """A single cell in a table with its content and position."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox_pdf: Optional[BBoxPDF] = None
    text: str = ""
    word_ids: List[int] = Field(default_factory=list)
    is_header: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TableQAMetrics(BaseModel):
    """Quality assurance metrics for table extraction."""
    coverage: float = 0.0  # % of words assigned to cells
    duplication_ratio: float = 0.0  # % of words assigned to multiple cells
    row_sanity_score: float = 1.0  # Consistency of column count across rows
    empty_cell_ratio: float = 0.0
    unassigned_word_ids: List[int] = Field(default_factory=list)
    passed: bool = True
    failure_reasons: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TableResult(BaseModel):
    """Complete result of table extraction."""
    table_id: str
    bbox_pdf: BBoxPDF
    table_type: TableType
    method: str  # "kv", "ruled", "tsr_tatr", "tsr_surya", etc.
    cells: List[TableCell] = Field(default_factory=list)
    qa: TableQAMetrics = Field(default_factory=TableQAMetrics)
    num_rows: int = 0
    num_cols: int = 0
    # Debug/provenance
    router_scores: Dict[str, float] = Field(default_factory=dict)
    extraction_time_ms: float = 0.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        # Enum serialization might need help if not auto-handled
        d = self.model_dump()
        d['table_type'] = self.table_type.value
        return d
    
    def to_markdown(self) -> str:
        """Render the table as Markdown."""
        if not self.cells:
            return ""
        
        # Build grid
        grid = {}
        for cell in self.cells:
            grid[(cell.row, cell.col)] = cell.text
        
        if not grid:
            return ""
        
        max_row = max(r for r, c in grid.keys())
        max_col = max(c for r, c in grid.keys())
        
        lines = []
        for row in range(max_row + 1):
            row_cells = [grid.get((row, col), "") for col in range(max_col + 1)]
            lines.append("| " + " | ".join(row_cells) + " |")
            if row == 0:
                lines.append("|" + "|".join(["---"] * (max_col + 1)) + "|")
        
        return "\n".join(lines)
