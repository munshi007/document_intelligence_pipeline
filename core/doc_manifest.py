"""
Document Manifest: Lossless Parallel Structure
Each element in the Markdown has a matching entry in this manifest,
preserving bounding boxes, font info, confidence, and provenance metadata
that Markdown cannot encode.

Design Reference: Docling's dual-output philosophy (MD for readability, JSON for lossless data).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ElementType(str, Enum):
    """Semantic element types for the manifest."""
    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    LIST_ITEM = "list_item"


class FontInfo(BaseModel):
    """Typography metadata for traceability."""
    size: Optional[float] = None
    fontname: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    color: Optional[str] = None


class ManifestElement(BaseModel):
    """
    A single element in the document manifest.
    Maps 1:1 to a region in the Markdown via `element_id`.
    """
    element_id: str                         # e.g., "p1_r0" — matches HTML comment in MD
    page: int
    element_type: ElementType
    bbox: List[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    text_preview: str = ""                  # First 120 chars for quick lookup
    confidence: float = 1.0
    source: str = "unknown"                 # "native_pdf", "layout_model", "ocr_repaired"
    font: Optional[FontInfo] = None

    # Relationships (populated by GraphBuilder)
    parent_element_id: Optional[str] = None   # Heading this element belongs to
    linked_element_ids: List[str] = Field(default_factory=list)  # Cross-refs (Fig X, Table Y)

    # Rich payload (only for tables/figures)
    table_data: Optional[Dict[str, Any]] = None
    figure_path: Optional[str] = None
    anchor_text: Optional[str] = None         # Caption or anchor text for tables


class DocumentManifest(BaseModel):
    """
    The lossless companion to the Markdown output.
    Every element_id referenced in Markdown HTML comments
    has a full entry here.
    """
    doc_id: str
    filename: str
    total_pages: int
    elements: List[ManifestElement] = Field(default_factory=list)

    def get_element(self, element_id: str) -> Optional[ManifestElement]:
        """Lookup an element by ID."""
        for el in self.elements:
            if el.element_id == element_id:
                return el
        return None

    def get_elements_by_page(self, page: int) -> List[ManifestElement]:
        """Get all elements on a specific page."""
        return [el for el in self.elements if el.page == page]

    def get_elements_by_type(self, etype: ElementType) -> List[ManifestElement]:
        """Get all elements of a given type."""
        return [el for el in self.elements if el.element_type == etype]

    def get_tables(self) -> List[ManifestElement]:
        return self.get_elements_by_type(ElementType.TABLE)

    def get_figures(self) -> List[ManifestElement]:
        return self.get_elements_by_type(ElementType.FIGURE)
