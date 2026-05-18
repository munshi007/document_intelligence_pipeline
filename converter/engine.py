"""
Converter Engine: Vision-to-Primal Graph (Librarian Edition)
Wraps the layout detection and OCR logic to produce:
  1. A list of standardized LayoutRegion objects (for GraphBuilder)
  2. A structured Markdown string with embedded provenance (HTML comments)
  3. A DocumentManifest (lossless JSON companion)

Design References:
  - Marker: Reading-order model to assign sequence indices
  - Docling: Dual output (Markdown + JSON manifest)
  - Unstructured: Element-level metadata enrichment
"""

import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz

from core.schemas import LayoutRegion, RegionType
from core.doc_manifest import (
    DocumentManifest, ManifestElement, ElementType, FontInfo
)
from pipeline.enhanced_pipeline import EnhancedPipeline

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────
def _map_region_type(raw_type: str) -> RegionType:
    """Robust fuzzy mapping from any YOLO/LayoutLMv3 label to RegionType."""
    raw = str(raw_type).strip().lower()
    try:
        return RegionType(raw)
    except ValueError:
        pass
    if any(x in raw for x in ['title', 'main_title', 'doc_title']):
        return RegionType.TITLE
    if any(x in raw for x in ['heading', 'sub_heading', 'section', 'h1', 'h2', 'h3']):
        return RegionType.HEADING
    if any(x in raw for x in ['table', 'spreadsheet', 'data_grid']):
        return RegionType.TABLE
    if any(x in raw for x in ['figure', 'image', 'picture', 'graphic', 'chart']):
        return RegionType.FIGURE
    if any(x in raw for x in ['caption', 'legend', 'fig_note']):
        return RegionType.CAPTION
    if any(x in raw for x in ['footer', 'page_footer', 'page_number']):
        return RegionType.FOOTER
    if any(x in raw for x in ['header', 'page_header', 'running_title']):
        return RegionType.HEADER
    return RegionType.TEXT


def _map_element_type(rt: RegionType) -> ElementType:
    """Map core RegionType → manifest ElementType."""
    mapping = {
        RegionType.TITLE:   ElementType.TITLE,
        RegionType.HEADING: ElementType.HEADING,
        RegionType.TEXT:    ElementType.TEXT,
        RegionType.TABLE:   ElementType.TABLE,
        RegionType.FIGURE:  ElementType.FIGURE,
        RegionType.CAPTION: ElementType.CAPTION,
        RegionType.FOOTER:  ElementType.FOOTER,
        RegionType.HEADER:  ElementType.HEADER,
    }
    return mapping.get(rt, ElementType.TEXT)


def _detect_heading_from_font(region_dict: dict) -> Optional[RegionType]:
    """
    Infer if a 'text' region is actually a heading/title based on font signature.
    Uses the font-size heuristic: bold text >= 12pt is likely a heading.
    """
    fs = region_dict.get('font_signature')
    if not fs:
        return None
    
    # Handle both dict and Pydantic object access
    def get_attr(obj, name, default):
        if hasattr(obj, name): return getattr(obj, name)
        if isinstance(obj, dict): return obj.get(name, default)
        return default

    size = get_attr(fs, 'size', 0)
    is_bold = get_attr(fs, 'is_bold', False)
    text = (region_dict.get('text') or '').strip()
    if not text:
        return None

    # Title: large bold text (>= 14pt)
    if is_bold and size >= 14:
        return RegionType.TITLE
    # Heading: bold text >= 11pt or ALL-CAPS text
    if is_bold and size >= 11:
        return RegionType.HEADING
    # ALL-CAPS short text is typically a heading
    if text.isupper() and len(text) > 3 and len(text) < 80:
        return RegionType.HEADING
    return None


def _sort_reading_order(regions: List[dict], page_width: float) -> List[dict]:
    """
    Geometry-first reading-order sort.
    Strategy:
      1. Divide the page into vertical 'lanes' (left/right columns).
      2. Within each lane, sort top-to-bottom.
      3. Interleave lanes left-to-right.
    For single-column documents this collapses to a simple top-to-bottom sort.
    """
    if not regions:
        return regions

    mid_x = page_width / 2
    # Classify each region as left-column or right-column
    left = []
    right = []
    full_width = []

    for r in regions:
        bbox = r.get('bbox', [0, 0, 0, 0])
        if len(bbox) < 4:
            full_width.append(r)
            continue
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        # If the region spans > 60% of page width, it's full-width
        if w > page_width * 0.60:
            full_width.append(r)
        elif (x1 + x2) / 2 < mid_x:
            left.append(r)
        else:
            right.append(r)

    # Sort each group top-to-bottom
    key = lambda r: r.get('bbox', [0, 0, 0, 0])[1]
    left.sort(key=key)
    right.sort(key=key)
    full_width.sort(key=key)

    # Interleave: full-width items go at their y-position, columns interleave
    # Simple merge: left column first, then right, with full-width inserted by y
    column_items = left + right
    column_items.sort(key=key)
    all_items = column_items + full_width
    all_items.sort(key=key)
    return all_items


# ── Main Engine ──────────────────────────────────────────────────────
class ConverterEngine:
    """
    High-level engine that uses the existing pipeline to extract primal regions.
    Produces:
      - List[LayoutRegion] for GraphBuilder
      - Structured Markdown with provenance comments
      - DocumentManifest (lossless JSON)
    """

    def __init__(self, output_dir: str = "output/v3", debug: bool = False, vlm_model: Optional[str] = None, vlm_provider: Optional[str] = None):
        # EnhancedPipeline is lazy: only the heavy stages (convert_to_regions
        # via self.pipeline.process_pdf, and any direct callers of
        # self.pipeline.components) trigger model loading. Pure helpers like
        # build_markdown_and_manifest() don't touch self.pipeline at all, so a
        # caller can construct a ConverterEngine and run them for free.
        self.output_dir = Path(output_dir)
        self._debug = debug
        self._vlm_model = vlm_model
        self._vlm_provider = vlm_provider
        self._pipeline: Optional["EnhancedPipeline"] = None

    @property
    def pipeline(self) -> "EnhancedPipeline":
        if self._pipeline is None:
            self._pipeline = EnhancedPipeline(
                output_dir=str(self.output_dir),
                debug_mode=self._debug,
                vlm_model=self._vlm_model,
                vlm_provider=self._vlm_provider,
            )
        return self._pipeline

    # ── public API ───────────────────────────────────────────────────
    def convert_to_regions(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
    ) -> List[LayoutRegion]:
        """Process a PDF and return a flat list of LayoutRegion objects."""
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Converting PDF to Primal Regions: {pdf_path}")
        doc_result = self.pipeline.process_pdf(str(pdf_path_obj), max_pages=max_pages)

        all_regions: List[LayoutRegion] = []

        for p_idx, page_data in enumerate(doc_result.get('pages', [])):
            page_num = page_data.get('page_num', p_idx + 1)
            page_w = page_data.get('page_size', {}).get('width', 600)
            raw_regions = page_data.get('regions', [])

            # 1. Reading-order sort
            sorted_regions = _sort_reading_order(raw_regions, page_w)

            for r_idx, r in enumerate(sorted_regions):
                region_type = _map_region_type(r.get('type', 'text'))

                # 2. Font-based heading promotion
                if region_type == RegionType.TEXT:
                    promoted = _detect_heading_from_font(r)
                    if promoted:
                        region_type = promoted

                region_obj = LayoutRegion(
                    id=f"p{page_num}_r{r_idx}",
                    page=page_num,
                    type=region_type,
                    bbox=r.get('bbox', [0, 0, 0, 0]),
                    text=r.get('text', ''),
                    confidence=r.get('confidence', 1.0),
                    source=r.get('source', 'pipeline'),
                    metadata=r,   # Preserve all raw data
                )
                all_regions.append(region_obj)

        return all_regions

    # ── Markdown + Manifest ──────────────────────────────────────────
    def build_markdown_and_manifest(
        self,
        regions: List[LayoutRegion],
        doc_info: Dict[str, Any],
    ) -> Tuple[str, DocumentManifest]:
        """
        Build a structured Markdown document WITH HTML-comment provenance
        and its companion DocumentManifest.
        """
        manifest = DocumentManifest(
            doc_id=doc_info.get('doc_id', 'unknown'),
            filename=doc_info.get('filename', 'unknown'),
            total_pages=doc_info.get('total_pages', 0),
        )

        md_lines: List[str] = []
        current_page = -1

        for r in regions:
            # Page break marker
            if r.page != current_page:
                if current_page != -1:
                    md_lines.append("\n---\n")
                md_lines.append(f"<!-- page:{r.page} -->")
                current_page = r.page

            # Build the provenance comment
            bbox_str = ",".join(f"{v:.1f}" for v in r.bbox)
            prov = f"<!-- id:{r.id} page:{r.page} type:{r.type.value} bbox:[{bbox_str}] src:{r.source} conf:{r.confidence:.2f} -->"

            # Build the Markdown line(s) for this region
            text = (r.text or "").strip()

            if r.type == RegionType.TITLE:
                md_lines.append(f"\n# {text}")
                md_lines.append(prov)

            elif r.type == RegionType.HEADING:
                md_lines.append(f"\n## {text}")
                md_lines.append(prov)

            elif r.type == RegionType.TABLE:
                # Reconstruct table from table_data
                table_data = r.metadata.get('table_data', {})
                rows = table_data.get('rows', [])
                anchor = r.metadata.get('anchor_text', '')
                if anchor:
                    md_lines.append(f"\n**{anchor}**")

                if rows:
                    headers = [str(c) if c else "" for c in rows[0]]
                    md_lines.append("\n| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows[1:]:
                        cells = [str(c) if c else "" for c in row]
                        md_lines.append("| " + " | ".join(cells) + " |")
                else:
                    md_lines.append(f"\n{text}")

                md_lines.append(prov)

            elif r.type == RegionType.FIGURE:
                fig_path = r.metadata.get('snapshot_image', '')
                md_lines.append(f"\n![Figure]({fig_path})")
                if text:
                    md_lines.append(f"_{text}_")
                md_lines.append(prov)

            elif r.type == RegionType.CAPTION:
                md_lines.append(f"\n_{text}_")
                md_lines.append(prov)

            elif r.type == RegionType.FOOTER:
                md_lines.append(f"\n<sub>{text}</sub>")
                md_lines.append(prov)

            else:
                # Normal text paragraph
                md_lines.append(f"\n{text}")
                md_lines.append(prov)

            # Build manifest element
            fs = r.metadata.get('font_signature')
            
            # Helper for safe access
            def get_attr(obj, name, default=None):
                if obj is None: return default
                if hasattr(obj, name): return getattr(obj, name)
                if isinstance(obj, dict): return obj.get(name, default)
                return default

            manifest.elements.append(ManifestElement(
                element_id=r.id,
                page=r.page,
                element_type=_map_element_type(r.type),
                bbox=r.bbox,
                text_preview=text[:120],
                confidence=r.confidence,
                source=r.source,
                font=FontInfo(
                    size=get_attr(fs, 'size'),
                    fontname=get_attr(fs, 'fontname'),
                    is_bold=get_attr(fs, 'is_bold', False),
                    is_italic=get_attr(fs, 'is_italic', False),
                    color=get_attr(fs, 'color'),
                ) if fs else None,
                table_data=r.metadata.get('table_data'),
                figure_path=r.metadata.get('snapshot_image'),
                anchor_text=r.metadata.get('anchor_text'),
            ))

        markdown_str = "\n".join(md_lines)
        return markdown_str, manifest
