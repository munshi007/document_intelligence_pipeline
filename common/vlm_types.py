"""
VLM Structured Output Types.
Defines Pydantic models for structured interaction with Vision Language Models.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

class ReadingOrderPrior(BaseModel):
    """
    Reading Order Planner output: High-level page structure analysis from a low-res image.
    Used to decide between simple xy-cut, column-first xy-cut, or complex deep learning.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis of the page structure. Are there columns? floating images?"
    )
    layout_type: Literal["simple_linear", "multi_column", "complex_unstructured"] = Field(
        description="Classification of the document layout."
    )
    suggested_strategy: Literal["xy_cut", "xy_cut_column_first", "deep_model"] = Field(
        description="The recommended reading order extraction strategy."
    )

class ReadingOrderVerification(BaseModel):
    """
    Reading Order Referee output: Quality assurance assessment of the extracted text flow.
    """
    reasoning: str = Field(
        description="Analysis of the text flow. Check if paragraphs are cut off or columns are interleaved."
    )
    is_continuous: bool = Field(
        description="Does the text flow logically?"
    )
    score: int = Field(
        ge=0, le=10, 
        description="Coherence score 0-10. 10 is perfect flow, 0 is jumbled soup."
    )
    suggested_action: Literal["accept", "rerun_column_first", "escalate"] = Field(
        description="Action to take based on the quality assessment."
    )


class TablePrior(BaseModel):
    """
    Planner output: High-level table structure analysis from a low-resolution image.
    Used to decide the best extraction strategy.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis of the table structure. Describe lines, alignment, and density."
    )
    is_table: bool = Field(
        description="Is this image region actually a table? False if it's just text or a figure."
    )
    table_type: Literal["ruled", "kv", "complex", "sparse"] = Field(
        description="Classification of the table type based on visual cues."
    )
    suggested_strategy: Literal["ruled_vector", "text_cluster", "hybrid", "complex_ltr"] = Field(
        description="Best extraction algorithm to use based on the table type."
    )

class TableVerification(BaseModel):
    """
    Referee output: Quality assurance assessment of the extracted data.
    Used to decide if re-extraction is needed.
    """
    reasoning: str = Field(
        description="Analysis of the extraction quality compared to the image. point out specific errors if any."
    )
    is_perfect: bool = Field(
        description="Is the extraction 100% correct matching the image?"
    )
    missing_rows: bool = Field(
        description="Are there obvious missing rows in the extraction?"
    )
    merged_columns: bool = Field(
        description="Are separate columns incorrectly merged in the text?"
    )
    score: int = Field(
        ge=0, le=10, 
        description="Quality score 0-10. 10 is perfect, 0 is garbage."
    )
    suggested_action: Literal["accept", "rerun_kv", "rerun_ruled", "rerun_complex", "escalate"] = Field(
        description="Action to take based on the quality assessment."
    )
class FontSignature(BaseModel):
    """Raw physical truth extracted from pdfplumber per text span."""
    size: float = Field(description="Font size in points (e.g., 14.5)")
    fontname: str = Field(description="Font name (e.g., 'TimesNewRomanPS-BoldMT')")
    is_bold: bool = Field(description="Derived from font flags")
    is_italic: bool = Field(description="Derived from font flags")
    color: Optional[str] = Field(default=None, description="Hex color (some docs use color for headers)")

    def __hash__(self):
        # We need this to be hashable for the deterministic fallback frequency counters
        return hash((round(self.size, 1), self.fontname, self.is_bold, self.is_italic, self.color))

    def __eq__(self, other):
        if not isinstance(other, FontSignature):
            return False
        return (round(self.size, 1), self.fontname, self.is_bold, self.is_italic, self.color) == \
               (round(other.size, 1), other.fontname, other.is_bold, other.is_italic, other.color)


class DocumentStyleSheet(BaseModel):
    """VLM-hypothesized stylesheet, grounded against physical signals."""
    reasoning: str = Field(description="Chain-of-thought analysis for these style assignments.")
    title: Optional[FontSignature] = None
    h1: Optional[FontSignature] = None
    h2: Optional[FontSignature] = None
    h3: Optional[FontSignature] = None
    body: FontSignature
    caption: Optional[FontSignature] = None

class RefinedRegion(BaseModel):
    """
    SOTA: Sub-Pixel Refinement output.
    Allows a high-res VLM glance to 'snap' a fuzzy detection to exact content boundaries.
    """
    reasoning: str = Field(description="Spatial reasoning for the refinement.")
    refined_bbox: list[float] = Field(description="Exact [x1, y1, x2, y2] in normalized 1000x1000 coordinates.")
    label: str = Field(description="Corrected semantic label if necessary.")
    confidence: float = Field(ge=0.0, le=1.0, description="Verification confidence.")

class VisualGapAnalysis(BaseModel):
    """
    SOTA: Visual Gap Analysis output.
    Used to scan 'empty' areas of a page for missed content (like small captions or logos).
    """
    found_missed_content: bool = Field(description="True if there is content in this crop not covered by existing boxes.")
    missed_regions: list[RefinedRegion] = Field(default_factory=list, description="List of newly discovered regions.")
