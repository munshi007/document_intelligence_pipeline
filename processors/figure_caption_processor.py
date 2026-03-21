"""
Figure and Caption Processor - Associate figures with captions using model detections
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class FigureCaptionProcessor:
    """
    Associate figures with captions using ensemble layout model detections and spatial analysis.
    
    Model-first approach:
    - Primary: Use ensemble detections (DocLayout-YOLO + LayoutParser) for Figure/FigureCaption
    - Secondary: Spatial proximity for association
    - Tertiary: Pattern matching only for validation/fallback
    
    The ensemble provides robust detection across different document types.
    """
    
    def __init__(self):
        """Initialize figure-caption processor"""
        # Caption patterns for validation only (not primary detection)
        self.caption_patterns = [
            r'^Figure\s+\d+',
            r'^Fig\.\s*\d+',
            r'^Table\s+\d+',
            r'^Diagram\s+\d+',
            r'^Chart\s+\d+'
        ]
    
    def associate_captions(
        self,
        regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Associate figure/table regions with their captions using model detections.
        
        Strategy:
        1. Use layout model's FigureCaption/TableCaption detections (primary)
        2. Spatial proximity for association
        3. Pattern matching only for validation
        
        Args:
            regions: List of all detected regions from layout model
            doc_profile: Optional document profile for adaptive thresholds
            
        Returns:
            Regions with caption associations added
        """
        if not regions:
            return regions
        
        # Separate figures, tables, and captions based on MODEL detections
        figures = [r for r in regions if r.get('type', '').lower() in ['figure']]
        tables = [r for r in regions if r.get('type', '').lower() in ['table']]
        figure_captions = [r for r in regions if r.get('type', '').lower() in ['figurecaption']]
        table_captions = [r for r in regions if r.get('type', '').lower() in ['tablecaption', 'tablefootnote']]
        
        logger.info(f"Model detected: {len(figures)} figures, {len(figure_captions)} figure captions")
        logger.info(f"Model detected: {len(tables)} tables, {len(table_captions)} table captions")
        
        # Associate figure captions with figures
        for figure in figures:
            caption = self._find_nearest_caption(figure, figure_captions, doc_profile)
            if caption:
                figure['caption'] = caption.get('text', '')
                figure['caption_bbox'] = caption.get('bbox')
                figure['caption_confidence'] = caption.get('confidence', 0.0)
                # Mark caption as associated
                caption['associated_with'] = figure.get('region_id')
        
        # Associate table captions with tables
        for table in tables:
            caption = self._find_nearest_caption(table, table_captions, doc_profile)
            if caption:
                table['caption'] = caption.get('text', '')
                table['caption_bbox'] = caption.get('bbox')
                table['caption_confidence'] = caption.get('confidence', 0.0)
                # Mark caption as associated
                caption['associated_with'] = table.get('region_id')
        
        # Fallback: Find captions that model missed using pattern matching
        text_regions = [r for r in regions if r.get('type', '').lower() in ['text', 'paragraph']]
        self._find_missed_captions(figures, tables, text_regions, doc_profile)
        
        return regions
    
    def _find_nearest_caption(
        self,
        parent_region: Dict[str, Any],
        caption_candidates: List[Dict[str, Any]],
        doc_profile: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the nearest caption to a figure/table using spatial proximity.
        
        Args:
            parent_region: Figure or table region
            caption_candidates: List of caption regions from model
            doc_profile: Optional document profile
            
        Returns:
            Nearest caption region or None
        """
        if not caption_candidates:
            return None
        
        # Get search distance (adaptive based on document)
        if doc_profile and hasattr(doc_profile, 'spacing_stats') and doc_profile.spacing_stats is not None:
            search_distance = doc_profile.spacing_stats.line_height * 3
        else:
            search_distance = 50  # Default pixels
        
        parent_bbox = parent_region['bbox']
        parent_bottom = parent_bbox[3]
        parent_top = parent_bbox[1]
        parent_center_x = (parent_bbox[0] + parent_bbox[2]) / 2
        
        # Find captions within search distance (prefer below, but check above too)
        nearby_captions = []
        
        for caption in caption_candidates:
            # Skip if already associated
            if caption.get('associated_with'):
                continue
            
            caption_bbox = caption['bbox']
            caption_top = caption_bbox[1]
            caption_bottom = caption_bbox[3]
            caption_center_x = (caption_bbox[0] + caption_bbox[2]) / 2
            
            # Check vertical distance (prefer captions below figure)
            vertical_dist_below = caption_top - parent_bottom
            vertical_dist_above = parent_top - caption_bottom
            
            # Check horizontal alignment
            horizontal_dist = abs(caption_center_x - parent_center_x)
            
            # Caption should be reasonably aligned horizontally
            if horizontal_dist < 200:  # Reasonable alignment threshold
                if 0 <= vertical_dist_below <= search_distance:
                    # Caption below (preferred)
                    nearby_captions.append((caption, vertical_dist_below, 'below'))
                elif 0 <= vertical_dist_above <= search_distance:
                    # Caption above (less common but possible)
                    nearby_captions.append((caption, vertical_dist_above, 'above'))
        
        if not nearby_captions:
            return None
        
        # Return closest caption (prefer below over above)
        nearby_captions.sort(key=lambda x: (0 if x[2] == 'below' else 1, x[1]))
        return nearby_captions[0][0]
    
    def _find_missed_captions(
        self,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        text_regions: List[Dict[str, Any]],
        doc_profile: Optional[Any]
    ):
        """
        Fallback: Find captions that the model missed using pattern matching.
        Only used when model doesn't detect caption regions.
        
        Args:
            figures: List of figure regions
            tables: List of table regions
            text_regions: List of text regions to search
            doc_profile: Optional document profile
        """
        # Only process figures/tables that don't have captions yet
        figures_without_captions = [f for f in figures if not f.get('caption')]
        tables_without_captions = [t for t in tables if not t.get('caption')]
        
        if not figures_without_captions and not tables_without_captions:
            return
        
        logger.info(f"Searching for missed captions using pattern matching...")
        
        # Search text regions for caption patterns
        for text_region in text_regions:
            text = (text_region.get('text') or '').strip()
            
            # Check if text matches caption pattern
            if self._matches_caption_pattern(text):
                # Determine if it's a figure or table caption
                is_figure_caption = any(p in text.lower() for p in ['figure', 'fig.', 'diagram', 'chart'])
                is_table_caption = 'table' in text.lower()
                
                if is_figure_caption and figures_without_captions:
                    # Find nearest figure
                    nearest = self._find_nearest_parent(text_region, figures_without_captions, doc_profile)
                    if nearest:
                        nearest['caption'] = text
                        nearest['caption_bbox'] = text_region['bbox']
                        nearest['caption_source'] = 'pattern_fallback'
                        figures_without_captions.remove(nearest)
                
                elif is_table_caption and tables_without_captions:
                    # Find nearest table
                    nearest = self._find_nearest_parent(text_region, tables_without_captions, doc_profile)
                    if nearest:
                        nearest['caption'] = text
                        nearest['caption_bbox'] = text_region['bbox']
                        nearest['caption_source'] = 'pattern_fallback'
                        tables_without_captions.remove(nearest)
    
    def _matches_caption_pattern(self, text: str) -> bool:
        """
        Check if text matches caption pattern (validation only).
        
        Args:
            text: Text to check
            
        Returns:
            True if matches caption pattern
        """
        for pattern in self.caption_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def _find_nearest_parent(
        self,
        caption_region: Dict[str, Any],
        parent_candidates: List[Dict[str, Any]],
        doc_profile: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the nearest figure/table for a caption region.
        
        Args:
            caption_region: Caption text region
            parent_candidates: List of figure/table regions
            doc_profile: Optional document profile
            
        Returns:
            Nearest parent region or None
        """
        if not parent_candidates:
            return None
        
        caption_bbox = caption_region['bbox']
        caption_center_y = (caption_bbox[1] + caption_bbox[3]) / 2
        caption_center_x = (caption_bbox[0] + caption_bbox[2]) / 2
        
        # Find closest parent
        min_distance = float('inf')
        nearest_parent = None
        
        for parent in parent_candidates:
            parent_bbox = parent['bbox']
            parent_center_y = (parent_bbox[1] + parent_bbox[3]) / 2
            parent_center_x = (parent_bbox[0] + parent_bbox[2]) / 2
            
            # Compute distance
            distance = ((caption_center_x - parent_center_x) ** 2 + 
                       (caption_center_y - parent_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_parent = parent
        
        # Only return if reasonably close (within 200 pixels)
        if min_distance < 200:
            return nearest_parent
        
        return None
