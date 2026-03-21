"""
Visualization Manager Module
Handles creation of debug visualizations and consolidated layout PDFs.
"""

import logging
import fitz
import numpy as np
from pathlib import Path
from typing import List, Dict

from utils.pipeline_utils import (
    create_bounding_box_visualization,
    create_debug_comparison,
    save_image
)

logger = logging.getLogger(__name__)

def create_bounding_box_visualization_wrapper(page_image: np.ndarray, regions: List[Dict], page_num: int, thumbnails_dir: Path, debug_mode: bool = False):
    """Create and save enhanced bounding box visualization."""
    try:
        # Use utility function to create visualization
        vis_image = create_bounding_box_visualization(page_image.copy(), regions, debug_mode=debug_mode)
        
        # Save visualization
        bbox_filename = f"page_{page_num:02d}_bboxes.png"
        bbox_path = thumbnails_dir / bbox_filename
        save_image(vis_image, bbox_path, f"enhanced bounding box visualization: {bbox_filename}")
        
        return vis_image
        
    except Exception as e:
        logger.warning(f"Failed to create bounding box visualization for page {page_num}: {e}")
        return None

def create_debug_visualizations(page_image: np.ndarray, regions: List[Dict], page_num: int, thumbnails_dir: Path, debug_dir: Path, debug_mode: bool = False):
    """
    Create debug visualizations using utility functions.
    Consolidates duplicate methods from original pipeline.py.
    """
    try:
        # Always create bounding box visualization
        create_bounding_box_visualization_wrapper(page_image, regions, page_num, thumbnails_dir, debug_mode=debug_mode)
        
        # Create debug comparison only in debug mode
        if debug_mode:
            # Create annotated version
            annotated = create_bounding_box_visualization(page_image.copy(), regions, debug_mode=True)
            
            # Create comparison using utility function
            debug_comparison = create_debug_comparison(page_image, annotated, regions)
            
            # Save debug comparison
            debug_filename = f"page_{page_num:02d}_debug.png"
            debug_path = debug_dir / debug_filename
            save_image(debug_comparison, debug_path, f"debug visualization: {debug_filename}")
            
    except Exception as e:
        logger.warning(f"Failed to create debug visualizations for page {page_num}: {e}")

def generate_layout_pdf(output_dir: Path, thumbnails_dir: Path):
    """Generate a single PDF containing all layout visualizations for easy review."""
    try:
        logger.info("Generating consolidated layout PDF...")
        layout_pdf_path = output_dir / "layout_visualizations.pdf"
        
        # Gather all debug/thumbnail images
        image_paths = sorted(list(thumbnails_dir.glob("*_bboxes.png")))
        if not image_paths:
            image_paths = sorted(list(thumbnails_dir.glob("page_*.png")))
        
        if not image_paths:
            logger.warning("No layout images found to consolidate.")
            return

        # Create PDF from images
        layout_doc = fitz.open()
        for img_path in image_paths:
            img = fitz.open(str(img_path))
            try:
                rect = img[0].rect
                pdfbytes = img.convert_to_pdf()
                img.close()
                imgPDF = fitz.open("pdf", pdfbytes)
                page = layout_doc.new_page(width=rect.width, height=rect.height)
                page.show_pdf_page(rect, imgPDF, 0)
            except Exception as e:
                logger.warning(f"Failed to add image {img_path} to PDF: {e}")
                img.close()
        
        layout_doc.save(str(layout_pdf_path))
        logger.info(f"Consolidated layout PDF saved to: {layout_pdf_path}")
        
    except Exception as e:
        logger.error(f"Error generating layout PDF: {e}")
