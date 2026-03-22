"""
PDF Primitives Extractor - Extract words and vector drawings from PDF pages.

Uses PyMuPDF (fitz) to extract raw primitives in PDF coordinate space.
All coordinates are in points (1/72 inch), with origin at top-left (fitz convention).
"""

import logging
from typing import Optional

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    fitz = None

from .types import (
    BBoxPDF,
    WordSpan,
    DrawingPrimitive,
    TablePrimitives,
    PageInfo,
)

logger = logging.getLogger(__name__)


class PdfPrimitivesExtractor:
    """
    Extract words and vector drawings from a PDF page using PyMuPDF.
    
    All outputs are in PDF coordinate space (points, top-left origin).
    Stores page transformation metadata for later coordinate conversions.
    """
    
    def __init__(self):
        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PdfPrimitivesExtractor. "
                "Install with: pip install pymupdf"
            )
    
    def extract(self, page: "fitz.Page") -> TablePrimitives:
        """
        Extract all primitives from a PDF page.
        
        Args:
            page: A fitz.Page object
            
        Returns:
            TablePrimitives containing words, drawings, and page metadata
        """
        words = self._extract_words(page)
        drawings = self._extract_drawings(page)
        images = self._extract_images(page)
        page_info = self._extract_page_info(page)
        
        logger.debug(
            f"Extracted {len(words)} words, {len(drawings)} drawings, {len(images)} images from page"
        )
        
        return TablePrimitives(
            words=words,
            drawings=drawings,
            images=images,
            page_info=page_info,
        )
    
    def _extract_words(self, page: "fitz.Page") -> list[WordSpan]:
        """
        Extract words with bounding boxes and font information.
        
        Uses page.get_text("dict") for rich metadata including font size.
        Falls back to page.get_text("words") if dict extraction fails.
        """
        words = []
        word_id = 0
        
        try:
            # Use dict extraction for rich metadata (font size, font name)
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            for block_no, block in enumerate(text_dict.get("blocks", [])):
                if block.get("type") != 0:  # Skip non-text blocks (images)
                    continue
                
                for line_no, line in enumerate(block.get("lines", [])):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        font_size = span.get("size", 0.0)
                        font_name = span.get("font", "")
                        
                        # Split span into individual words for finer granularity
                        span_words = text.split()
                        if not span_words:
                            continue
                        
                        # For single-word spans, use the span bbox directly
                        if len(span_words) == 1:
                            words.append(WordSpan(
                                id=word_id,
                                text=text,
                                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                block_no=block_no,
                                line_no=line_no,
                                word_no=0,
                                font_size=font_size,
                                font_name=font_name,
                            ))
                            word_id += 1
                        else:
                            # For multi-word spans, estimate word positions
                            # This is approximate; for precise word boxes use "words" mode
                            span_width = bbox[2] - bbox[0]
                            total_chars = len(text)
                            x_pos = bbox[0]
                            
                            for word_no, word in enumerate(span_words):
                                word_width = (len(word) / total_chars) * span_width if total_chars > 0 else span_width
                                word_bbox = (x_pos, bbox[1], x_pos + word_width, bbox[3])
                                
                                words.append(WordSpan(
                                    id=word_id,
                                    text=word,
                                    bbox=word_bbox,
                                    block_no=block_no,
                                    line_no=line_no,
                                    word_no=word_no,
                                    font_size=font_size,
                                    font_name=font_name,
                                ))
                                word_id += 1
                                x_pos += word_width + (span_width * 0.02)  # Small gap between words
                                
        except Exception as e:
            logger.warning(f"Dict extraction failed, falling back to words mode: {e}")
            words = self._extract_words_simple(page)
        
        return words
    
    def _extract_words_simple(self, page: "fitz.Page") -> list[WordSpan]:
        """
        Fallback word extraction using simple words mode.
        
        Returns words with bboxes but without font metadata.
        """
        words = []
        raw_words = page.get_text("words")
        
        for word_id, w in enumerate(raw_words):
            # w = (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            if len(w) < 5:
                continue
            
            text = str(w[4]).strip()
            if not text:
                continue
            
            words.append(WordSpan(
                id=word_id,
                text=text,
                bbox=(w[0], w[1], w[2], w[3]),
                block_no=int(w[5]) if len(w) > 5 else 0,
                line_no=int(w[6]) if len(w) > 6 else 0,
                word_no=int(w[7]) if len(w) > 7 else 0,
                font_size=0.0,  # Not available in simple mode
                font_name="",
            ))
        
        return words
    
    def _extract_drawings(self, page: "fitz.Page") -> list[DrawingPrimitive]:
        """
        Extract vector drawings (lines and rectangles) from the page.
        
        Uses page.get_drawings() which returns vector path information.
        Filters to lines and rectangles useful for table detection.
        """
        drawings = []
        
        try:
            raw_drawings = page.get_drawings()
        except Exception as e:
            logger.warning(f"Failed to extract drawings: {e}")
            return drawings
        
        for draw in raw_drawings:
            # Each drawing has 'items' which are path commands
            items = draw.get("items", [])
            rect = draw.get("rect")  # Bounding rectangle
            color = draw.get("color")
            width = draw.get("width")
            if width is None:
                width = 1.0
            
            if not rect:
                continue
            
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            
            # Analyze items to classify as line or rect
            for item in items:
                cmd = item[0]  # Command type: 'l' (line), 're' (rect), 'c' (curve), etc.
                
                if cmd == "l":  # Line
                    # item = ('l', p1, p2)
                    if len(item) >= 3:
                        p1 = (item[1].x, item[1].y) if hasattr(item[1], 'x') else item[1]
                        p2 = (item[2].x, item[2].y) if hasattr(item[2], 'x') else item[2]
                        
                        drawings.append(DrawingPrimitive(
                            kind="line",
                            bbox=bbox,
                            points=[p1, p2],
                            width=width,
                            color=color,
                        ))
                
                elif cmd == "re":  # Rectangle
                    drawings.append(DrawingPrimitive(
                        kind="rect",
                        bbox=bbox,
                        points=[],
                        width=width,
                        color=color,
                    ))
        
        # Also extract rectangles from page annotations if any
        # (Some PDFs use annotations for table borders)
        
        
        return drawings
    
    def _extract_images(self, page: "fitz.Page") -> list:
        """Extract embedded raster image bounding boxes."""
        from .types import ImagePrimitive
        images = []
        try:
            for img in page.get_images():
                xref = img[0]
                rects = page.get_image_rects(xref)
                for rect in rects:
                    images.append(ImagePrimitive(
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        width=img[2],
                        height=img[3]
                    ))
        except Exception as e:
            logger.warning(f"Failed to extract images: {e}")
        return images
    
    def _extract_page_info(self, page: "fitz.Page") -> PageInfo:
        """Extract page metadata for coordinate transformations."""
        rect = page.rect
        cropbox = page.cropbox
        mediabox = page.mediabox
        
        return PageInfo(
            rotation=page.rotation,
            cropbox=(cropbox.x0, cropbox.y0, cropbox.x1, cropbox.y1) if cropbox else None,
            mediabox=(mediabox.x0, mediabox.y0, mediabox.x1, mediabox.y1) if mediabox else None,
            width=rect.width,
            height=rect.height,
            transform_matrix=tuple(page.transformation_matrix) if page.transformation_matrix else None,
        )
    
    def extract_from_pdf(self, pdf_path: str, page_num: int) -> TablePrimitives:
        """
        Convenience method to extract primitives from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            TablePrimitives for the specified page
        """
        doc = fitz.open(pdf_path)
        try:
            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} does not exist (PDF has {len(doc)} pages)")
            page = doc[page_num]
            return self.extract(page)
        finally:
            doc.close()
