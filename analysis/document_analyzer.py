"""
Document Analyzer - Compute adaptive statistics and thresholds
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    FORM = "form"
    PRESENTATION = "presentation"
    UNKNOWN = "unknown"


@dataclass
class FontStats:
    """Font statistics from document"""
    mean_size: float
    std_size: float
    min_size: float
    max_size: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


@dataclass
class SpacingStats:
    """Spacing statistics from document"""
    mean_line_spacing: float
    std_line_spacing: float
    mean_paragraph_spacing: float
    line_height: float
    percentile_5: float
    percentile_95: float


@dataclass
class ResolutionStats:
    """Resolution and dimension statistics"""
    dpi: float
    width: float
    height: float
    aspect_ratio: float


@dataclass
class AdaptiveThresholds:
    """Computed adaptive thresholds for processing"""
    heading_font_size: float
    merge_distance: float
    column_gap: float
    confidence_threshold: float
    line_spacing: float
    paragraph_spacing: float


@dataclass
class DocumentProfile:
    """Complete document profile with statistics and thresholds"""
    resolution: ResolutionStats
    font_stats: Optional[FontStats]
    spacing_stats: Optional[SpacingStats]
    layout_density: float
    document_type: DocumentType
    thresholds: AdaptiveThresholds


class DocumentAnalyzer:
    """Analyze document to compute adaptive parameters"""
    
    def __init__(self):
        """Initialize document analyzer"""
        pass
    
    def analyze(self, page_image: np.ndarray, ocr_results: List[Dict[str, Any]] = None, layout_regions: List[Dict[str, Any]] = None) -> DocumentProfile:
        """
        Analyze document and compute adaptive profile
        
        Args:
            page_image: Page image as numpy array
            ocr_results: Optional OCR results with font information
            layout_regions: Optional layout detection results
            
        Returns:
            DocumentProfile with computed statistics and thresholds
        """
        # Compute resolution stats
        resolution = self._compute_resolution_stats(page_image)
        
        # Compute font stats if OCR results available
        font_stats = None
        if ocr_results:
            font_stats = self.compute_font_statistics(ocr_results)
        
        # Compute spacing stats if OCR results available
        spacing_stats = None
        if ocr_results:
            spacing_stats = self._compute_spacing_statistics(ocr_results)
        
        # Compute layout density
        layout_density = self._compute_layout_density(page_image)
        
        # Detect document type
        document_type = self.detect_document_type(
            layout_density, 
            font_stats, 
            spacing_stats, 
            layout_regions
        )
        
        # Compute adaptive thresholds
        thresholds = self.compute_adaptive_thresholds(font_stats, spacing_stats, resolution)
        
        return DocumentProfile(
            resolution=resolution,
            font_stats=font_stats,
            spacing_stats=spacing_stats,
            layout_density=layout_density,
            document_type=document_type,
            thresholds=thresholds
        )
    
    def compute_font_statistics(self, ocr_results: List[Dict[str, Any]]) -> FontStats:
        """
        Compute font statistics from OCR results
        
        Args:
            ocr_results: List of OCR results with bbox and optional font info
            
        Returns:
            FontStats with computed statistics
        """
        # Extract font sizes (use bbox height as proxy if font_size not available)
        font_sizes = []
        for result in ocr_results:
            if 'font_size' in result:
                font_sizes.append(result['font_size'])
            elif 'bbox' in result:
                bbox = result['bbox']
                height = bbox[3] - bbox[1]
                font_sizes.append(height)
        
        if not font_sizes:
            # Return default stats if no data
            return FontStats(
                mean_size=12.0,
                std_size=2.0,
                min_size=8.0,
                max_size=24.0,
                percentile_25=10.0,
                percentile_75=14.0,
                percentile_95=18.0
            )
        
        font_sizes = np.array(font_sizes)
        
        return FontStats(
            mean_size=float(np.mean(font_sizes)),
            std_size=float(np.std(font_sizes)),
            min_size=float(np.min(font_sizes)),
            max_size=float(np.max(font_sizes)),
            percentile_25=float(np.percentile(font_sizes, 25)),
            percentile_75=float(np.percentile(font_sizes, 75)),
            percentile_95=float(np.percentile(font_sizes, 95))
        )
    
    def _compute_spacing_statistics(self, ocr_results: List[Dict[str, Any]]) -> SpacingStats:
        """Compute spacing statistics from OCR results"""
        if not ocr_results or len(ocr_results) < 2:
            # Return defaults
            return SpacingStats(
                mean_line_spacing=15.0,
                std_line_spacing=5.0,
                mean_paragraph_spacing=30.0,
                line_height=12.0,
                percentile_5=10.0,
                percentile_95=25.0
            )
        
        # Sort by Y coordinate
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        # Compute vertical gaps between consecutive text blocks
        gaps = []
        line_heights = []
        
        for i in range(len(sorted_results) - 1):
            curr_bbox = sorted_results[i]['bbox']
            next_bbox = sorted_results[i + 1]['bbox']
            
            # Gap between bottom of current and top of next
            gap = next_bbox[1] - curr_bbox[3]
            if gap > 0:
                gaps.append(gap)
            
            # Line height
            line_heights.append(curr_bbox[3] - curr_bbox[1])
        
        if not gaps:
            gaps = [15.0]
        if not line_heights:
            line_heights = [12.0]
        
        gaps = np.array(gaps)
        line_heights = np.array(line_heights)
        
        return SpacingStats(
            mean_line_spacing=float(np.mean(gaps)),
            std_line_spacing=float(np.std(gaps)),
            mean_paragraph_spacing=float(np.percentile(gaps, 75)),
            line_height=float(np.mean(line_heights)),
            percentile_5=float(np.percentile(gaps, 5)),
            percentile_95=float(np.percentile(gaps, 95))
        )
    
    def _compute_resolution_stats(self, page_image: np.ndarray) -> ResolutionStats:
        """Compute resolution statistics from page image"""
        height, width = page_image.shape[:2]
        
        # Assume 300 DPI for now (can be refined)
        dpi = 300.0
        
        return ResolutionStats(
            dpi=dpi,
            width=float(width),
            height=float(height),
            aspect_ratio=float(width / height) if height > 0 else 1.0
        )
    
    def _compute_layout_density(self, page_image: np.ndarray) -> float:
        """Compute layout density (ratio of non-white pixels)"""
        try:
            import cv2
            gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
            # Threshold to binary
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            # Compute density
            density = np.sum(binary > 0) / binary.size
            return float(density)
        except Exception as e:
            logger.warning(f"Failed to compute layout density: {e}")
            return 0.1  # Default
    
    def compute_adaptive_thresholds(
        self, 
        font_stats: Optional[FontStats],
        spacing_stats: Optional[SpacingStats],
        resolution: ResolutionStats
    ) -> AdaptiveThresholds:
        """
        Compute adaptive thresholds from statistics
        
        Args:
            font_stats: Font statistics
            spacing_stats: Spacing statistics
            resolution: Resolution statistics
            
        Returns:
            AdaptiveThresholds with computed values
        """
        # Heading font size threshold: mean + 1.5*std
        if font_stats:
            heading_font_size = font_stats.mean_size + 1.5 * font_stats.std_size
        else:
            heading_font_size = 18.0
        
        # Merge distance: 5th percentile of line spacing
        if spacing_stats:
            merge_distance = spacing_stats.percentile_5
            line_spacing = spacing_stats.mean_line_spacing
            paragraph_spacing = spacing_stats.mean_paragraph_spacing
        else:
            merge_distance = 10.0
            line_spacing = 15.0
            paragraph_spacing = 30.0
        
        # Column gap: 95th percentile of horizontal gaps (placeholder for now)
        column_gap = resolution.width * 0.1  # 10% of page width
        
        # Confidence threshold: 25th percentile (placeholder)
        confidence_threshold = 0.5
        
        return AdaptiveThresholds(
            heading_font_size=heading_font_size,
            merge_distance=merge_distance,
            column_gap=column_gap,
            confidence_threshold=confidence_threshold,
            line_spacing=line_spacing,
            paragraph_spacing=paragraph_spacing
        )
    
    def detect_document_type(
        self,
        layout_density: float,
        font_stats: Optional[FontStats],
        spacing_stats: Optional[SpacingStats],
        layout_regions: Optional[List[Dict[str, Any]]] = None
    ) -> DocumentType:
        """
        Detect document type based on layout density and structure
        
        Args:
            layout_density: Ratio of non-white pixels
            font_stats: Font statistics
            spacing_stats: Spacing statistics
            layout_regions: Optional layout detection results
            
        Returns:
            DocumentType classification
        """
        try:
            # Count region types if available
            region_counts = {}
            if layout_regions:
                for region in layout_regions:
                    region_type = region.get('type', 'unknown')
                    region_counts[region_type] = region_counts.get(region_type, 0) + 1
            
            # Academic papers: high density, many text regions, figures, tables
            # Typically have title, abstract, references
            if layout_regions:
                has_title = region_counts.get('title', 0) > 0
                has_figures = region_counts.get('figure', 0) > 0
                has_tables = region_counts.get('table', 0) > 0
                text_count = region_counts.get('text', 0)
                
                # Academic: title + (figures or tables) + many text blocks
                if has_title and (has_figures or has_tables) and text_count > 5:
                    if layout_density > 0.15:  # Dense text
                        return DocumentType.ACADEMIC
                
                # Technical: similar to academic but may lack title
                if (has_figures or has_tables) and text_count > 3:
                    if layout_density > 0.12:
                        return DocumentType.TECHNICAL
                
                # Form: low text density, structured layout, many small regions
                if layout_density < 0.08 and len(layout_regions) > 10:
                    # Forms have many small regions with low text density
                    avg_region_size = sum(
                        (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1])
                        for r in layout_regions
                    ) / len(layout_regions)
                    
                    # If average region is small relative to page
                    if avg_region_size < 50000:  # Arbitrary threshold
                        return DocumentType.FORM
                
                # Presentation: low density, large fonts, few text blocks
                if text_count < 5 and layout_density < 0.10:
                    if font_stats and font_stats.mean_size > 16:
                        return DocumentType.PRESENTATION
            
            # Fallback: use density and font stats
            if layout_density > 0.15:
                # High density suggests academic or technical
                if font_stats and font_stats.std_size > 3:
                    # High variance in font sizes suggests academic (headings, body, captions)
                    return DocumentType.ACADEMIC
                else:
                    return DocumentType.TECHNICAL
            elif layout_density < 0.08:
                # Low density suggests form or presentation
                if font_stats and font_stats.mean_size > 16:
                    return DocumentType.PRESENTATION
                else:
                    return DocumentType.FORM
            else:
                # Medium density - likely technical document
                return DocumentType.TECHNICAL
                
        except Exception as e:
            logger.warning(f"Failed to detect document type: {e}")
            return DocumentType.UNKNOWN
