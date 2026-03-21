
import logging
import re
from typing import List, Dict, Any

import numpy as np

# Import configurable thresholds
try:
    from config import PROCESSING_CONFIG
except ImportError:
    PROCESSING_CONFIG = {
        'min_table_rows': 1,
        'min_table_cols': 1,
        'min_meaningful_cells': 2,
        'max_empty_cell_ratio': 0.8,
        'filter_placeholder_cells': True,
    }

logger = logging.getLogger(__name__)


class MarkdownRenderer:
    """
    Renders ordered regions to markdown with proper formatting for headings, paragraphs, tables, and figures.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.stylesheet = None

    def set_stylesheet(self, stylesheet: Any):
        """Set the document-wide stylesheet for consistent heading rendering."""
        self.stylesheet = stylesheet

    def extract_markdown_from_regions(self, regions: List[Dict[str, Any]]) -> str:
        """
        Convert ordered regions to markdown format.
        
        Args:
            regions: List of regions in reading order
            
        Returns:
            Markdown formatted string
        """
        parts: List[str] = []

        for region in regions:
            region_id = region.get("region_id", "unknown")
            region_type = region.get("type", region.get("region_type", "unknown"))
            bbox = region.get("bbox", [])
            method = region.get("method", region.get("source", "unknown"))

            # Render based on region type
            rendered = self._render_region(region)
            if rendered:
                parts.append(rendered)

        return "\n\n".join([p for p in parts if p is not None])
    
    def _render_region(self, region: Dict[str, Any]) -> str:
        """
        Render a single region based on its type.
        
        Args:
            region: Region dictionary
            
        Returns:
            Markdown formatted string
        """
        region_type = region.get("type", region.get("region_type", "unknown")).lower()
        
        # Headings
        if region_type in ["title", "heading"]:
            return self._render_heading(region)
        
        # Tables (case-insensitive)
        elif region_type.lower() == "table":
            return self._render_table(region)
        
        # Figures (case-insensitive)
        elif region_type.lower() == "figure":
            return self._render_figure(region)
        
        # Lists
        elif region.get("is_list_item"):
            return self._render_list_item(region)
        
        # Captions (if not already associated with figure/table)
        elif region_type in ["figurecaption", "tablecaption"] and not region.get("associated_with"):
            return self._render_caption(region)
        
        # Regular text/paragraphs
        else:
            text = region.get("text", "").strip()
            return text if text else None
    
    def _render_heading(self, region: Dict[str, Any]) -> str:
        """Render heading with appropriate level based on stylesheet or size."""
        text = region.get("text", "").strip()
        if not text:
            return None
        
        # SOTA: Prioritize grounded stylesheet
        if self.stylesheet:
            region_font = region.get("font_signature")
            if region_font:
                # Compare against stylesheet signatures
                if self.stylesheet.h1 and region_font == self.stylesheet.h1:
                    return f"# {text}"
                if self.stylesheet.h2 and region_font == self.stylesheet.h2:
                    return f"## {text}"
                if self.stylesheet.h3 and region_font == self.stylesheet.h3:
                    return f"### {text}"
                if self.stylesheet.title and region_font == self.stylesheet.title:
                    return f"# {text}"

        # FALLBACK: Heuristic logic
        font_size = region.get("font_size", 12)
        # If we have a font signature but no stylesheet match, use the size from signature
        if region.get("font_signature"):
            font_size = region["font_signature"].size

        region_type = region.get("type", "").lower()
        
        if region_type == "title" or font_size > 20:
            level = 1
        elif font_size > 16:
            level = 2
        else:
            level = 3
        
        return f"{'#' * level} {text}"
    
    def _render_list_item(self, region: Dict[str, Any]) -> str:
        """Render list item (already has bullet/number in text)."""
        text = region.get("text", "").strip()
        return text if text else None

    def _render_table(self, region: Dict[str, Any]) -> str:
        """Render table region to markdown."""
        # SOTA: Specialist Markdown Table (e.g. from GOT-OCR2.0)
        table_data = region.get("table_data", {})
        if table_data.get("markdown_table"):
            table_md = table_data["markdown_table"]
            logger.info(f"MarkdownRenderer: Using specialist markdown table for region {region.get('id')}")
        else:
            # Legacy/Deterministic row-based rendering
            rows = table_data.get("rows") or region.get("rows")
            
            if not rows or len(rows) == 0:
                return None
            
            # Validate table - filter garbage tables
            if not self._is_valid_table(rows):
                logger.debug(f"Filtering invalid/garbage table with {len(rows)} rows")
                return None
            
            table_md = self._format_table_as_markdown(rows)
        
        # Add caption if available
        caption = region.get("caption")
        if caption:
            return f"{caption}\n\n{table_md}"
        
        return table_md
    
    def _is_valid_table(self, rows: List[List[str]]) -> bool:
        """
        Validate if a table has meaningful content or is garbage.
        Filters out tables with:
        - Only placeholder cells (Cell_X_Y, Col1, Col2, etc.)
        - Too many empty cells
        - No meaningful content
        
        Args:
            rows: Table rows
            
        Returns:
            True if table is valid, False if garbage
        """
        if not rows:
            return False
        
        min_meaningful_cells = PROCESSING_CONFIG.get('min_meaningful_cells', 2)
        max_empty_ratio = PROCESSING_CONFIG.get('max_empty_cell_ratio', 0.8)
        filter_placeholders = PROCESSING_CONFIG.get('filter_placeholder_cells', True)
        
        # Patterns for garbage/placeholder cells
        placeholder_patterns = [
            r'^Cell_\d+_\d+$',      # Cell_0_0, Cell_16_0, etc.
            r'^Col\d+$',            # Col1, Col2, etc.
            r'^Row\d+$',            # Row1, Row2, etc.
            r'^Column\d+$',         # Column1, Column2, etc.
            r'^R\d+C\d+$',          # R1C1, R2C3, etc.
        ]
        placeholder_regex = re.compile('|'.join(placeholder_patterns), re.IGNORECASE)
        
        total_cells = 0
        empty_cells = 0
        placeholder_cells = 0
        meaningful_cells = 0
        
        for row in rows:
            for cell in row:
                total_cells += 1
                cell_str = str(cell).strip() if cell else ""
                
                if not cell_str:
                    empty_cells += 1
                elif filter_placeholders and placeholder_regex.match(cell_str):
                    placeholder_cells += 1
                else:
                    meaningful_cells += 1
        
        if total_cells == 0:
            return False
        
        # Check if table has enough meaningful content
        empty_ratio = (empty_cells + placeholder_cells) / total_cells
        
        if empty_ratio > max_empty_ratio:
            logger.debug(f"Table rejected: {empty_ratio:.1%} empty/placeholder cells")
            return False
        
        if meaningful_cells < min_meaningful_cells:
            logger.debug(f"Table rejected: only {meaningful_cells} meaningful cells")
            return False
        
        # Check if ALL non-empty cells are placeholders
        non_empty = total_cells - empty_cells
        if non_empty > 0 and placeholder_cells == non_empty:
            logger.debug("Table rejected: all cells are placeholders")
            return False
        
        return True
    
    def _render_figure(self, region: Dict[str, Any]) -> str:
        """Render figure with caption."""
        parts = []
        
        # Add figure image
        img_path = region.get("snapshot_image")
        if img_path:
            parts.append(f"![Figure]({img_path})")
        
        # Add caption
        caption = region.get("caption") or region.get("text")
        if caption:
            parts.append(caption)
        
        return "\n\n".join(parts) if parts else None
    
    def _render_caption(self, region: Dict[str, Any]) -> str:
        """Render standalone caption (not associated with figure/table)."""
        text = region.get("text", "").strip()
        if text:
            return f"*{text}*"  # Italicize captions
        return None
    
    def _format_table_as_markdown(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        lengths = [len(r) for r in rows if r]
        if not lengths:
            return ""
        median_len = int(np.median(lengths))
        if median_len <= 0:
            return ""

        norm: List[List[str]] = []
        for r in rows:
            r = r[:median_len] + [""] * (median_len - len(r))
            norm.append([self._clean_cell_for_markdown(c) for c in r])

        header_keywords = [
            "Pin", "Channel", "IN", "OUT", "Type", "Name", "Value", "Unit",
            "Description", "Min", "Max", "Typ", "Typical", "Condition", "Parameter",
            "Symbol", "Rating", "Limit", "Test", "No", "Number", "Voltage", "Current",
            "Frequency", "Temperature", "Power", "Input", "Output", "Function"
        ]

        def row_has_header_words(r: List[str]) -> bool:
            return any(any(k.lower() in (c or "").lower() for k in header_keywords) for c in r)

        if row_has_header_words(norm[0]):
            header = norm[0]
            data_rows = norm[1:]
        else:
            header = [f"Col{i+1}" for i in range(median_len)]
            data_rows = norm

        lines: List[str] = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * median_len) + " |")
        for r in data_rows:
            lines.append("| " + " | ".join(r) + " |")

        return "\n".join(lines)

    @staticmethod
    def _clean_cell_for_markdown(s: Any) -> str:
        if s is None:
            return ""
        text = str(s)
        text = text.replace("|", r"\|")
        return " ".join(text.split())
