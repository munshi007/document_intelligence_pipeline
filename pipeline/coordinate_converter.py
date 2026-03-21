"""
Coordinate Converter Module
Handles coordinate transformations between PDF and image space.
"""

import numpy as np
import cv2
import fitz
from typing import List, Dict, Tuple

def convert_page_to_image(page: fitz.Page, dpi: int = 200) -> Tuple[np.ndarray, float]:
    """
    Convert PDF page to numpy image array and return scaling info.
    
    Args:
        page: PyMuPDF Page object
        dpi: DPI for rendering (default 200)
        
    Returns:
        Tuple of (image_array, dpi_scale)
    """
    dpi_scale = dpi / 72
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
    
    return page_image, dpi_scale

def convert_regions_to_pdf_coords(regions: List[Dict], dpi_scale: float) -> List[Dict]:
    """
    Convert region coordinates from image space to PDF coordinate space.
    
    Args:
        regions: List of region dictionaries with 'bbox' in image space
        dpi_scale: Scale factor (DPI / 72)
        
    Returns:
        List of regions with 'bbox' in PDF space
    """
    converted_regions = []
    
    for region in regions:
        # Create a copy of the region
        converted_region = region.copy()
        
        # Convert bounding box coordinates
        if 'bbox' in region:
            bbox = region['bbox']
            converted_bbox = [
                bbox[0] / dpi_scale,  # x1
                bbox[1] / dpi_scale,  # y1
                bbox[2] / dpi_scale,  # x2
                bbox[3] / dpi_scale   # y2
            ]
            converted_region['bbox'] = converted_bbox
        
        converted_regions.append(converted_region)
    
    return converted_regions

def convert_regions_to_image_coords(regions: List[Dict], dpi_scale: float) -> List[Dict]:
    """
    Convert region coordinates from PDF space back to image coordinate space.
    
    Args:
        regions: List of region dictionaries with 'bbox' in PDF space
        dpi_scale: Scale factor (DPI / 72)
        
    Returns:
        List of regions with 'bbox' in image space
    """
    converted_regions = []
    
    for region in regions:
        # Create a copy of the region
        converted_region = region.copy()
        
        # Convert bounding box coordinates
        if 'bbox' in region:
            bbox = region['bbox']
            converted_bbox = [
                bbox[0] * dpi_scale,  # x1
                bbox[1] * dpi_scale,  # y1
                bbox[2] * dpi_scale,  # x2
                bbox[3] * dpi_scale   # y2
            ]
            converted_region['bbox'] = converted_bbox
        
        converted_regions.append(converted_region)
    
    return converted_regions
