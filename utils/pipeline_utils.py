"""
Utility functions for the Enhanced PDF Processing Pipeline
"""

import logging
import re
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def classify_text_content(text: str, font_info: Dict) -> str:
    """Classify text content based on font properties and text characteristics (exact copy from original)."""
    from config import FONT_CONFIG
    
    font_size = font_info.get("size", 12)
    font_flags = font_info.get("flags", 0)
    
    # Check for bold text (flag 16)
    is_bold = bool(font_flags & 16)
    
    # Classify based on font size and properties
    if font_size >= FONT_CONFIG['heading_font_threshold'] or is_bold:
        if len(text) < 100 and not text.endswith('.'):
            return "heading"
    
    # Check for list items
    if re.match(r'^\s*[•\-\*]\s+', text) or re.match(r'^\s*\d+[\.\)]\s+', text):
        return "list"
    
    # Check for captions
    if re.match(r'^(Figure|Table|Diagram|Chart)\s+\d+', text, re.IGNORECASE):
        return "caption"
    
    # Check for table-like content
    if '\t' in text or '|' in text or len(re.findall(r'\s{3,}', text)) > 2:
        return "table_text"
    
    # Default to paragraph
    return "paragraph"

def extract_text_blocks_with_fonts(page) -> List[Dict[str, Any]]:
    """
    Extract text blocks with font information from a PDF page.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        List of text blocks with font information
    """
    blocks = []
    
    try:
        # Get text blocks with font information
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    font_size = span.get("size", 12)
                    font_flags = span.get("flags", 0)
                    
                    # Classify text type (pass font_info like original)
                    font_info = {
                        "size": font_size,
                        "flags": font_flags,
                        "font": span.get("font", ""),
                    }
                    text_type = classify_text_content(text, font_info)
                    
                    blocks.append({
                        "text": text,
                        "bbox": list(bbox),
                        "font_info": {
                            "size": font_size,
                            "flags": font_flags,
                            "font": span.get("font", ""),
                        },
                        "type": text_type,
                        "source": "text_extraction"
                    })
                    
    except Exception as e:
        logger.error(f"Error extracting text blocks: {e}")
    
    return blocks

def save_image(image: np.ndarray, filepath: Path, description: str = "") -> bool:
    """
    Save image to file with error handling.
    
    Args:
        image: Image array
        filepath: Path to save the image
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Save using PIL for better compatibility
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(filepath)
        
        if description:
            logger.info(f"Saved {description}: {filepath.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving image {filepath}: {e}")
        return False

def aggregate_region_stats(regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics from regions.
    
    Args:
        regions: List of region dictionaries
        
    Returns:
        Dictionary with region type, processing method, and table statistics
    """
    region_types = {}
    processing_methods = {}
    
    # Count tables with actual table data
    tables_found = 0
    
    for region in regions:
        # Count region types
        region_type = region.get('type', 'unknown')
        region_types[region_type] = region_types.get(region_type, 0) + 1
        
        # Count processing methods
        source = region.get('source', 'unknown')
        processing_methods[source] = processing_methods.get(source, 0) + 1
        
        # Count tables with actual data
        if region_type in ['Table', 'table'] and region.get('table_data'):
            tables_found += 1
    
    return {
        'region_types': region_types,
        'processing_methods': processing_methods,
        'tables_found': tables_found
    }

def validate_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Validate and clamp bounding box coordinates to image bounds.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        image_width: Image width
        image_height: Image height
        
    Returns:
        Validated bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Clamp to image bounds
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = min(x1 + 1, image_width)
    if y2 <= y1:
        y2 = min(y1 + 1, image_height)
    
    return [float(x1), float(y1), float(x2), float(y2)]

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def format_confidence(confidence: float) -> str:
    """Format confidence score for display."""
    return f"{confidence:.3f}"

def get_region_area(bbox: List[float]) -> float:
    """Calculate area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def create_bounding_box_visualization(page_image: np.ndarray, regions: List[Dict], debug_mode: bool = False) -> np.ndarray:
    """Create bounding box visualization with clean labels (matching original behavior)."""
    try:
        # Create a copy of the image for drawing
        vis_image = page_image.copy()
        
        # Define colors for different region types (Case-insensitive mapping)
        colors_map = {
            'text': (0, 255, 0),      # Green
            'paragraph': (0, 255, 0), # Green (same as text for consistency)
            'title': (255, 0, 0),     # Red  
            'heading': (255, 0, 0),   # Red (same as title)
            'table': (0, 0, 255),     # Blue
            'figure': (255, 255, 0),  # Cyan
            'list': (255, 0, 255),    # Magenta
            'caption': (255, 165, 0), # Orange
            'table_text': (0, 100, 255), # Light Blue
        }
        
        # Draw bounding boxes with clean labels
        for i, region in enumerate(regions):
            region_type = region.get('type', 'Unknown')
            bbox = region['bbox']
            confidence = region.get('confidence', 0.0)
            source = region.get('source', 'unknown')
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = colors_map.get(region_type.lower(), (128, 128, 128))  # Default gray
            
            # Draw rectangle with thickness based on confidence
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Add clean label (only add debug info in debug mode)
            label = f"{i}: {region_type}"
            if debug_mode:
                label += f" ({confidence:.2f}, {source})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 8), 
                        (x1 + label_size[0] + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1 + 2, y1 - 4), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add text preview only in debug mode
            if debug_mode and region.get('text'):
                text_preview = region['text'][:30] + "..." if len(region['text']) > 30 else region['text']
                cv2.putText(vis_image, text_preview, (x1, y2 + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return vis_image
        
    except Exception as e:
        logger.error(f"Error creating bounding box visualization: {e}")
        return page_image

def create_debug_comparison(original: np.ndarray, annotated: np.ndarray, regions: List[Dict]) -> np.ndarray:
    """Create side-by-side debug comparison with region info."""
    try:
        height, width = original.shape[:2]
        
        # Create side-by-side image
        debug_image = np.zeros((height, width * 2 + 50, 3), dtype=np.uint8)
        
        # Place original and annotated images
        debug_image[:, :width] = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        debug_image[:, width + 50:] = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        # Add separator line
        cv2.line(debug_image, (width + 25, 0), (width + 25, height), (255, 255, 255), 2)
        
        # Add title
        cv2.putText(debug_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_image, "Detected Regions", (width + 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return debug_image
        
    except Exception as e:
        logger.error(f"Error creating debug comparison: {e}")
        return original

def assess_text_quality(text_blocks: List[Dict[str, Any]]) -> bool:
    """
    Assess the quality of extracted text chunks.
    Returns True if text appears valid (English), False if it looks like garbage/corrupted.
    """
    if not text_blocks:
        return False
        
    # Concatenate a sample of text
    full_text = " ".join([b.get('text', '') for b in text_blocks]) # Check ALL blocks to be sure
    
    if len(full_text) < 50:
        return True # Too little text to judge
        
    # Check for encoded garbage (Private Use Area characters common in bad PDFs)
    # Range E000-F8FF is Basic Multilingual Plane Private Use Area
    pua_count = sum(1 for c in full_text if 0xE000 <= ord(c) <= 0xF8FF)
    pua_ratio = pua_count / len(full_text)
    
    if pua_ratio > 0.01: # If > 1% characters are PUA, it's definitely garbage
        logger.warning(f"Low text quality: High PUA character ratio ({pua_ratio:.2%}). Fallback to OCR.")
        return False
        
    # Check for common English stop words
    # If text is English, these should appear frequently
    common_words = {'the', 'and', 'of', 'in', 'to', 'a', 'is', 'that', 'for', 'it', 'with', 'on', 'as', 'are', 'be', 'this', 'was', 'at', 'by', 'an'}
    
    # Normalize: remove non-alpha to avoid matching parts of words
    words = [w.lower() for w in re.findall(r'\\b[a-z]{2,}\\b', full_text)]
    total_words = len(words)
    
    if total_words < 10:
        return True # Not enough words to judge statistically
        
    common_count = sum(1 for w in words if w in common_words)
    ratio = common_count / total_words
    
    # Heuristic: Valid English text usually has > 20% stop words.
    if ratio < 0.15:
        logger.warning(f"Low text quality: Low common word ratio ({ratio:.2%}). Fallback to OCR.")
        return False
        
    return True
