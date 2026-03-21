"""
Region Cleaner Module
Handles cleanup, deduplication, and overlap checking for regions.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def is_contained(small_bbox: List[float], large_bbox: List[float], threshold: float = 0.8) -> bool:
    """
    Check if small_bbox is contained within large_bbox.
    
    Args:
        small_bbox: [x1, y1, x2, y2]
        large_bbox: [x1, y1, x2, y2]
        threshold: Overlap threshold (default 0.8)
        
    Returns:
        True if significantly contained
    """
    x1_min, y1_min, x1_max, y1_max = small_bbox
    x2_min, y2_min, x2_max, y2_max = large_bbox
    
    # Check if small bbox is mostly inside large bbox
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return False
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    small_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # If most of the small bbox is inside the large bbox
    return (inter_area / small_area) > threshold if small_area > 0 else False

def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def cleanup_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Smart cleanup: Trust model labels, only filter text that's truly inside tables/figures.
    
    Strategy:
    1. Keep ALL model-detected regions (they're semantic, not spatial)
    2. For PyMuPDF text: only keep if it's in "empty space" (not covered by model regions)
    3. Tables are handled by pdfplumber, so remove OCR text from tables.
    """
    # Separate by source
    model_regions = [r for r in regions if r.get('source') == 'layout_model']
    text_regions = [r for r in regions if r.get('source') != 'layout_model']
    
    logger.info(f"Cleanup: {len(model_regions)} model regions, {len(text_regions)} text regions")
    
    # For each text region, check if it's in "empty space"
    filtered_text = []
    for text_region in text_regions:
        text_bbox = text_region.get('bbox')
        if not text_bbox:
            continue
        
        # Check if this text is already covered by a model region
        is_duplicate = False
        for model_region in model_regions:
            model_type = model_region.get('type', '').lower()
            model_bbox = model_region.get('bbox')
            
            if not model_bbox:
                continue
            
            # SOTA: Filter text against ALL model regions (including Title/Text) to avoid duplication
            # If the model found a box, we trust its contents over a raw PDF block
            if is_contained(text_bbox, model_bbox, threshold=0.5):
                is_duplicate = True
                logger.debug(f"Removing native text overlapping with model {model_type} region")
                break
        
        # Additional text-based deduplication (prevent identical text from being added twice)
        if not is_duplicate:
            current_text = (text_region.get('text') or '').strip()
            if any(current_text == (r.get('text') or '').strip() for r in model_regions if r.get('text')):
                is_duplicate = True
                logger.debug(f"Removing identical text duplicate")
        
        if not is_duplicate:
            filtered_text.append(text_region)
    
    logger.info(f"After filtering: {len(filtered_text)} text regions kept")
    
    # Combine checks
    all_regions = model_regions + filtered_text
    
    return all_regions
