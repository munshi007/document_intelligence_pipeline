"""
Region Processor - Hierarchical processing that trusts model labels
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Import configurable thresholds
try:
    from config import PROCESSING_CONFIG
except ImportError:
    PROCESSING_CONFIG = {
        'figure_overlap_threshold': 0.75,
        'table_overlap_threshold': 0.4,
        'base_overlap_threshold': 0.5,
    }

logger = logging.getLogger(__name__)


class RegionProcessor:
    """
    Process regions hierarchically, trusting layout model's semantic labels.
    Filters out text that's inside figures/tables.
    Uses LayoutLMv3 for smart classification when needed.
    """
    
    def __init__(self, use_layoutlm: bool = True):
        """
        Initialize region processor.
        
        Args:
            use_layoutlm: Whether to use LayoutLMv3 for smart classification
        """
        self.use_layoutlm = use_layoutlm
        self.layoutlm_classifier = None
        
        if use_layoutlm:
            try:
                from modules.layoutlm_classifier import LayoutLMClassifier
                self.layoutlm_classifier = LayoutLMClassifier(use_layoutlm=True)
                logger.info("LayoutLM classifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LayoutLM classifier: {e}")
                self.layoutlm_classifier = None
    
    def process_regions_hierarchically(
        self,
        layout_regions: List[Dict[str, Any]],
        text_regions: List[Dict[str, Any]],
        page_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Process regions in priority order, filtering text inside figures/tables.
        
        Priority:
        1. Figure/Table (from model) - mark as occupied space
        2. FigureCaption/TableCaption - associate with parent
        3. Title/Heading - extract as headings
        4. Text regions - extract as paragraphs
        5. PyMuPDF text - only from unoccupied space
        
        Args:
            layout_regions: Regions from layout model
            text_regions: Text regions from PyMuPDF
            page_image: Page image for LayoutLMv3 (optional)
            
        Returns:
            Processed and filtered regions
        """
        # Separate regions by type
        figures = [r for r in layout_regions if r.get('type', '').lower() in ['figure']]
        tables = [r for r in layout_regions if r.get('type', '').lower() in ['table']]
        captions = [r for r in layout_regions if 'caption' in r.get('type', '').lower()]
        titles = [r for r in layout_regions if r.get('type', '').lower() in ['title']]
        model_text = [r for r in layout_regions if r.get('type', '').lower() in ['text']]
        
        logger.info(f"Hierarchical processing: {len(figures)} figures, {len(tables)} tables, "
                   f"{len(captions)} captions, {len(titles)} titles, {len(model_text)} model text")
        
        # Mark occupied space (figures, tables, titles, and model-detected text)
        # SOTA: We include ALL layout regions to avoid duplication with native blocks
        occupied_bboxes = figures + tables + titles + model_text + captions
        
        # Filter text regions that are inside occupied space
        # Use LayoutLMv3 for smart filtering if available
        filtered_text = self._filter_text_inside_regions(
            text_regions, 
            occupied_bboxes,
            page_image=page_image
        )
        logger.info(f"Filtered {len(text_regions) - len(filtered_text)} text regions inside figures/tables")
        
        # Classify ambiguous text regions using LayoutLMv3
        if self.layoutlm_classifier and page_image is not None:
            filtered_text = self._classify_text_regions(filtered_text, page_image)
            logger.info("Applied LayoutLMv3 classification to text regions")
        
        # Combine in priority order
        processed_regions = []
        
        # Add figures (with figure_internal flag)
        for figure in figures:
            figure['is_figure'] = True
            processed_regions.append(figure)
        
        # Add tables
        for table in tables:
            table['is_table'] = True
            processed_regions.append(table)
        
        # Add captions
        processed_regions.extend(captions)
        
        # Add titles
        processed_regions.extend(titles)
        
        # Add model text regions
        processed_regions.extend(model_text)
        
        # Add filtered PyMuPDF text
        processed_regions.extend(filtered_text)
        
        logger.info(f"Hierarchical processing complete: {len(processed_regions)} regions")
        return processed_regions
    
    def _classify_text_regions(
        self,
        text_regions: List[Dict[str, Any]],
        page_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Classify text regions using LayoutLMv3 for ambiguous cases.
        
        Args:
            text_regions: Text regions to classify
            page_image: Page image
            
        Returns:
            Text regions with updated classifications
        """
        for region in text_regions:
            text = region.get('text', '')
            bbox = region.get('bbox')
            
            # Only classify ambiguous cases (short text, no clear type)
            if len(text.strip()) < 20 and not region.get('type'):
                classification = self.layoutlm_classifier.classify_text_region(
                    text, bbox, page_image
                )
                
                # Update region type based on classification
                if classification == "label":
                    region['type'] = 'label'
                elif classification == "heading":
                    region['type'] = 'heading'
                elif classification == "caption":
                    region['type'] = 'caption'
                else:
                    region['type'] = 'text'
                
                logger.debug(f"Classified '{text[:30]}...' as {classification}")
        
        return text_regions
    
    def _compute_intersection_ratio(self, text_bbox: List[float], region_bbox: List[float]) -> float:
        """Compute ratio of text area that is inside the region."""
        x1_min, y1_min, x1_max, y1_max = text_bbox
        x2_min, y2_min, x2_max, y2_max = region_bbox
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        text_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        if text_area == 0:
            return 0.0
            
        return inter_area / text_area

    def _filter_text_inside_regions(
        self,
        text_regions: List[Dict[str, Any]],
        occupied_regions: List[Dict[str, Any]],
        threshold: float = 0.5,
        page_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter out text regions that are inside figures/tables.
        Uses different thresholds for figures vs tables.
        Uses LayoutLMv3 for smart relationship detection when available.
        
        Args:
            text_regions: Text regions from PyMuPDF
            occupied_regions: Figure/Table regions
            threshold: Base intersection ratio threshold
            page_image: Page image for LayoutLMv3 (optional)
            
        Returns:
            Filtered text regions
        """
        if not occupied_regions:
            return text_regions
        
        # Use configurable thresholds from config
        FIGURE_THRESHOLD = PROCESSING_CONFIG.get('figure_overlap_threshold', 0.75)
        TABLE_THRESHOLD = PROCESSING_CONFIG.get('table_overlap_threshold', 0.4)
        
        filtered = []
        
        for text_region in text_regions:
            text_bbox = text_region.get('bbox')
            if not text_bbox:
                continue
            
            # Check if text is inside ANY occupied region (or combination of them)
            # We sum the intersection areas to handle text spanning multiple regions (e.g. split tables)
            total_intersection_area = 0.0
            text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
            
            if text_area <= 0:
                continue
                
            is_inside = False
            
            # First pass: Check individual regions for LayoutLMv3 relationships
            for occupied in occupied_regions:
                occupied_bbox = occupied.get('bbox')
                if not occupied_bbox:
                    continue
                
                # Use different threshold based on region type
                region_type = occupied.get('type', '').lower()
                if 'figure' in region_type:
                    effective_threshold = FIGURE_THRESHOLD
                elif 'table' in region_type:
                    effective_threshold = TABLE_THRESHOLD
                else:
                    effective_threshold = threshold
                
                # Calculate intersection with this region
                x1_min, y1_min, x1_max, y1_max = text_bbox
                x2_min, y2_min, x2_max, y2_max = occupied_bbox
                
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                
                if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    total_intersection_area += inter_area
                    
                    # Check for specific relationships if using LayoutLMv3
                    intersection_ratio = inter_area / text_area
                    if intersection_ratio > effective_threshold and self.layoutlm_classifier and page_image is not None:
                        relationship = self.layoutlm_classifier.detect_relationship(
                            text_region, [occupied], page_image
                        )
                        if relationship == "caption":
                            text_region['type'] = 'caption'
                            text_region['associated_figure'] = occupied.get('id')
                            logger.debug(f"LayoutLMv3: Keeping text as caption")
                            # Don't count this as "inside" for filtering purposes if it's a caption
                            # But we might still filter it if it overlaps heavily with other things?
                            # For now, let's assume captions are kept.
                            is_inside = False 
                            # Reset total intersection to avoid filtering? 
                            # No, if it's a caption, we want to keep it.
                            # So we should break and keep it.
                            break
            
            # If it was identified as a caption, keep it
            if text_region.get('type') == 'caption':
                filtered.append(text_region)
                continue

            # Check total intersection ratio against base threshold
            total_intersection_ratio = total_intersection_area / text_area
            
            if total_intersection_ratio > threshold:
                is_inside = True
                logger.debug(f"Spatial: Excluding text '{text_region.get('text', '')[:30]}...' "
                           f"(Total Intersection={total_intersection_ratio:.2f})")
            
            if not is_inside:
                filtered.append(text_region)
        
        return filtered
