"""
Page Processor Module
Handles processing of a single PDF page, orchestrating layout detection, 
OCR, and region processing.
"""

import logging
import fitz
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from config import OUTPUT_CONFIG
from processors.region_merger import merge_regions
from processors.page_processor import PageProcessor as SnapshotProcessor
from utils.pipeline_utils import save_image

from pipeline.coordinate_converter import (
    convert_page_to_image,
    convert_regions_to_pdf_coords,
    convert_regions_to_image_coords
)
from pipeline.region_cleaner import cleanup_regions
from pipeline.visualization_manager import create_debug_visualizations, create_bounding_box_visualization_wrapper
from common.types import BBox

logger = logging.getLogger(__name__)

class PageProcessor:
    """
    Handles processing logic for a single page.
    Encapsulates the 'process_page' logic from the original pipeline.
    """
    
    def __init__(self, components: Dict[str, Any], output_paths: Dict[str, Any], debug_mode: bool = False):
        """
        Initialize PageProcessor with shared components and output paths.
        """
        self.components = components
        self.output_paths = output_paths
        self.debug_mode = debug_mode
        
        # Unpack directories
        self.thumbnails_dir = output_paths['thumbnails']
        self.tables_dir = output_paths['tables']
        self.figures_dir = output_paths['figures']
        self.debug_dir = output_paths['debug']
        
        # Unpack components
        self.document_analyzer = components.get('document_analyzer')
        self.layout_detector = components.get('layout_detector')
        self.ocr_engine = components.get('ocr_engine')
        self.reading_order_resolver = components.get('reading_order_resolver')
        self.figure_caption_processor = components.get('figure_caption_processor')
        self.region_processor = components.get('region_processor')
        self.table_extractor = components.get('table_extractor')
        self.markdown_renderer = components.get('markdown_renderer')
        self.snapshot_processor = components.get('page_processor')
        self.vlm_client = components.get('vlm_client')
        
    def process_page(self, page: fitz.Page, page_num: int, current_pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF page using initialized components."""
        logger.info(f"Processing page {page_num}")
        
        try:
            # Metadata initialization
            page_info = {
                "page_num": page_num,
                "page_size": {"width": page.rect.width, "height": page.rect.height}
            }
            vlm_metadata = {
                "image_id": f"{Path(current_pdf_path).stem}_p{page_num:03d}",
                "pdf_path": current_pdf_path,
                "page_num": page_num
            }
            
            # Convert page to image
            page_image, dpi_scale = convert_page_to_image(page)
            
            # Save thumbnail
            thumbnail_path = self.thumbnails_dir / f"page_{page_num:02d}.png"
            save_image(page_image, thumbnail_path, f"page thumbnail: page_{page_num:02d}.png")
            page_info["thumbnail_path"] = f"{OUTPUT_CONFIG['thumbnails_subdir']}/page_{page_num:02d}.png"
            
            # Step 0: Document Analysis
            doc_profile = self.document_analyzer.analyze(page_image)
            
            # Step 1: Detect layout regions (RT-DETR Specialist)
            layout_regions = self.layout_detector.detect_layout_regions(page_image, debug=self.debug_mode)
            
            # Step 2: Native Text Extraction (for orphan recovery)
            text_blocks_pdf = []
            for i, block in enumerate(page.get_text("blocks")):
                text_blocks_pdf.append({
                    "type": "text", "bbox": [block[0], block[1], block[2], block[3]],
                    "text": block[4], "source": "native_pdf", "confidence": 1.0,
                    "id": f"native_{page_num}_{i}"
                })
            text_blocks_image = convert_regions_to_image_coords(text_blocks_pdf, dpi_scale)

            # Step 3: Hierarchical Region Processing
            layout_regions = self.region_processor.process_regions_hierarchically(
                layout_regions, text_blocks_image, page_image=page_image
            )

            # Step 4: Figure-Caption Association
            layout_regions = self.figure_caption_processor.associate_captions(layout_regions, doc_profile)

            # Step 5: Table Extraction (Tables v2 Coordinator)
            for i, table_region in enumerate([r for r in layout_regions if r.get('type') == 'Table']):
                image_bbox = table_region['bbox']
                pdf_bbox = [coord / dpi_scale for coord in image_bbox]
                
                table_data = self.table_extractor.extract_table_structure(
                    page_image, image_bbox, page_num, i+1,
                    doc_profile=doc_profile, pdf_path=current_pdf_path,
                    pdf_page_num=page_num - 1, pdf_bbox=pdf_bbox,
                    fitz_page=page, vlm_metadata=vlm_metadata
                )
                table_region['table_data'] = table_data
                table_region['table_image_path'] = f"table_page_{page_num:02d}_{i+1:02d}.png"
            
            # Standard visualizations
            create_bounding_box_visualization_wrapper(page_image, layout_regions, page_num, self.thumbnails_dir, self.debug_mode)
            
            # Snapshots
            if self.snapshot_processor:
                layout_regions = self.snapshot_processor.attach_region_snapshots(page_image, layout_regions)
            
            # Coordinate conversion & Cleanup
            merged_regions = convert_regions_to_pdf_coords(layout_regions, dpi_scale)
            merged_regions = merge_regions(merged_regions, [])
            merged_regions = cleanup_regions(merged_regions)
            
            # Final Native Text Extraction
            for r in merged_regions:
                if not (r.get('text') or '').strip():
                    bbox = r.get('bbox')
                    if bbox:
                        try:
                            rect = fitz.Rect(bbox)
                            r['text'] = page.get_textbox(rect) or page.get_text("text", clip=rect)
                        except: pass

            # Reading Order (XY-Cut)
            merged_regions = self.reading_order_resolver.order_regions(merged_regions, page_image, doc_profile)
            
            # Markdown Generation
            self.markdown_renderer.clean_output = not self.debug_mode
            markdown_content = self.markdown_renderer.extract_markdown_from_regions(merged_regions)

            # Debug visualizations
            if self.debug_mode:
                image_space_regions = convert_regions_to_image_coords(merged_regions, dpi_scale)
                create_debug_visualizations(page_image, image_space_regions, page_num, self.thumbnails_dir, self.debug_dir, self.debug_mode)
            
            # Final Page Result
            page_result = {
                **page_info,
                "regions": merged_regions,
                "markdown": markdown_content,
                "stats": {
                    "total_regions": len(merged_regions),
                    "tables_found": len([r for r in layout_regions if r.get('type') == 'Table']),
                    "figures_found": len([r for r in layout_regions if r.get('type') == 'Figure'])
                }
            }
            
            logger.info(f"Page {page_num} processed successfully")
            return page_result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return {"page_num": page_num, "error": str(e), "regions": [], "markdown": "", "stats": {}}
