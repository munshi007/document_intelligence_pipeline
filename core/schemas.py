from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RegionType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class LayoutRegion(BaseModel):
    """Primal structural unit recognized by the Vision system."""
    id: str
    page: int
    type: RegionType
    bbox: List[float]
    text: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Relationships for Graph-Native logic
    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)

class TableData(BaseModel):
    rows: List[List[str]]
    num_rows: int
    num_cols: int
    caption: Optional[str] = None
    method: str = "unknown"

class HierarchicalNode(BaseModel):
    """A semantic object in the HKG (Hierarchical Knowledge Graph)."""
    node_id: str
    type: str # e.g., "section", "hybrid_chunk"
    hierarchy: List[str] # ["Introduction", "Specifications"]
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Graph Links (SOTA 2026)
    linked_nodes: List[str] = Field(default_factory=list) # Cross-references (See Fig. X)
    regions: List[LayoutRegion] = Field(default_factory=list)

class DocumentGraph(BaseModel):
    """The full 'Structural Fact Store' (HKG) for a document."""
    doc_id: str
    filename: str
    nodes: List[HierarchicalNode] = Field(default_factory=list)
    total_pages: int
    
class Extraction(BaseModel):
    """A single extracted entity/fact grounded in source nodes."""
    extraction_class: str
    extraction_text: str
    source_node_ids: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)

class ExtractionResult(BaseModel):
    """Final output for a target schema."""
    schema_title: str
    data: List[Extraction] = Field(default_factory=list)
    confidence_score: float = 0.0
    reasoning: Optional[str] = None
