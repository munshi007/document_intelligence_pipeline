"""
Font Extraction Engine
Uses pdfplumber to extract physical font stats (size, name, flags, color)
and provides a deterministic fallback to infer a DocumentStyleSheet.
"""

import pdfplumber
import logging
from typing import List, Dict, Any
from collections import Counter

from common.types import LayoutRegion
from common.vlm_types import FontSignature, DocumentStyleSheet

logger = logging.getLogger(__name__)

class FontAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_fonts_for_page(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extracts all character spans with their font properties for a specific page.
        Returns a list of dicts: {'text': str, 'bbox': [x0, y0, x1, y1], 'font_signature': FontSignature}
        """
        chars_info = []
        with pdfplumber.open(self.pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                logger.error(f"Page number {page_num} out of bounds.")
                return chars_info
                
            page = pdf.pages[page_num - 1]
            words = page.extract_words(
                extra_attrs=["fontname", "size"],
                keep_blank_chars=False
            )
            
            for word in words:
                fontname = word.get('fontname', 'Unknown')
                size = float(word.get('size', 0.0))
                is_bold = 'bold' in fontname.lower() or 'black' in fontname.lower() or 'heavy' in fontname.lower()
                is_italic = 'italic' in fontname.lower() or 'oblique' in fontname.lower()
                
                color = None
                raw_color = word.get('color')
                if raw_color:
                    if isinstance(raw_color, (tuple, list)):
                        try:
                            # If it's a 0-1 range tuple for RGB
                            if all(isinstance(c, float) and c <= 1.0 for c in raw_color):
                                color = "#{:02x}{:02x}{:02x}".format(
                                    int(raw_color[0]*255), 
                                    int(raw_color[1]*255), 
                                    int(raw_color[2]*255)
                                )
                        except Exception:
                            pass
                
                fs = FontSignature(
                    size=size,
                    fontname=fontname,
                    is_bold=is_bold,
                    is_italic=is_italic,
                    color=color
                )
                
                chars_info.append({
                    'text': word.get('text', ''),
                    'bbox': [word['x0'], word['top'], word['x1'], word['bottom']],
                    'font_signature': fs,
                })
                
        return chars_info

    def assign_fonts_to_regions(self, regions: List[Any], page_num: int) -> List[Any]:
        """
        Aggregates stats per bounding box and assigns the dominant font size.
        Supports both LayoutRegion objects and standard dictionaries.
        """
        chars_info = self.extract_fonts_for_page(page_num)
        if not chars_info:
            return regions
            
        for region in regions:
            # Handle both dict and object
            is_dict = isinstance(region, dict)
            rtype = region.get('type', '') if is_dict else getattr(region, 'type', '')
            
            # We only care about text-like regions for fonts (e.g., Title, Text, List, heading)
            if rtype.lower() not in ['text', 'title', 'list', 'header', 'footer', 'heading', 'paragraph']:
                continue
                
            # Extract bbox coordinates
            if is_dict:
                bbox_raw = region.get('bbox', [0,0,0,0])
                rx0, ry0, rx1, ry1 = bbox_raw
            else:
                rx0, ry0, rx1, ry1 = region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1
            
            # Find all words that fall inside this region
            region_fonts = []
            for char_info in chars_info:
                cx0, cy0, cx1, cy1 = char_info['bbox']
                
                # Check if char bbox center is inside region bbox
                center_x = (cx0 + cx1) / 2
                center_y = (cy0 + cy1) / 2
                
                if (rx0 <= center_x <= rx1) and (ry0 <= center_y <= ry1):
                    region_fonts.append(char_info['font_signature'])
                    
            if region_fonts:
                dominant_font = Counter(region_fonts).most_common(1)[0][0]
                if is_dict:
                    region['font_signature'] = dominant_font
                    # Also set font_size as a flat float for legacy renderer compatibility
                    region['font_size'] = dominant_font.size
                else:
                    region.font_signature = dominant_font
                
        return regions

    def infer_stylesheet_from_stats(self, page_limit: int = 3) -> DocumentStyleSheet:
        """
        Step 4: Deterministic Fallback.
        Extracts all unique (size, is_bold) pairs from the first N pages.
        Uses frequency analysis and size ranking to build a stylesheet without VLM.
        """
        all_fonts = []
        with pdfplumber.open(self.pdf_path) as pdf:
            max_pages = min(page_limit, len(pdf.pages))
            for i in range(max_pages):
                page = pdf.pages[i]
                words = page.extract_words(extra_attrs=["fontname", "size"])
                for word in words:
                    size = float(word.get('size', 0.0))
                    fontname = word.get('fontname', '')
                    is_bold = 'bold' in fontname.lower() or 'black' in fontname.lower()
                    fs = FontSignature(
                        size=size,
                        fontname=fontname,
                        is_bold=is_bold,
                        is_italic='italic' in fontname.lower(),
                        color=None # Ignore color for statistical fallback to stay robust
                    )
                    all_fonts.append(fs)
                    
        if not all_fonts:
            # Fallback if text extraction fails entirely (e.g. scanned PDF)
            return DocumentStyleSheet(
                reasoning="Fallback: No extractable text found, using default 12pt body.",
                body=FontSignature(size=12.0, fontname="Unknown", is_bold=False, is_italic=False, color=None)
            )
            
        # The most frequent font signature is highly likely to be the body text
        font_counts = Counter(all_fonts)
        
        # We prefer un-bold text for body if possible
        body_candidates = [fs for fs in font_counts.keys() if not fs.is_bold]
        if body_candidates:
            body_font = max(body_candidates, key=lambda fs: font_counts[fs])
        else:
            body_font = font_counts.most_common(1)[0][0]
            
        # Find heading candidates: anything larger than body, or same size but bold
        heading_candidates = set()
        for fs in font_counts.keys():
            if fs.size > body_font.size + 0.5: # Clearly larger
                heading_candidates.add(fs)
            elif abs(fs.size - body_font.size) <= 0.5 and fs.is_bold and not body_font.is_bold:
                # Same size but bold
                heading_candidates.add(fs)
                
        # Sort candidates by size descending, then by boldness
        sorted_candidates = sorted(list(heading_candidates), key=lambda fs: (fs.size, fs.is_bold), reverse=True)
        
        stylesheet = DocumentStyleSheet(
            reasoning="Deterministic Fallback: Built purely from text span statistics over the first few pages.",
            body=body_font
        )
        
        # Assign H1, H2, H3 based on sorted order
        if len(sorted_candidates) > 0:
            stylesheet.h1 = sorted_candidates[0]
        if len(sorted_candidates) > 1:
            stylesheet.h2 = sorted_candidates[1]
        if len(sorted_candidates) > 2:
            stylesheet.h3 = sorted_candidates[2]
            
        return stylesheet
