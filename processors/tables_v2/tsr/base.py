"""
TSR (Table Structure Recognition) Engine Interface

Defines the abstract interface for TSR models (TATR, TableFormer, Surya, etc.)
and coordinate mapping utilities for pixel -> PDF conversion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from ..types import BBoxPDF, TableCell


@dataclass
class CellPx:
    """A cell detected by TSR model in pixel coordinates (relative to crop)."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox_px: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (x0, y0, x1, y1) in pixels
    confidence: float = 1.0
    is_header: bool = False


class TSREngine(ABC):
    """
    Abstract base class for Table Structure Recognition engines.
    
    Implementations must provide predict_cells() which takes a cropped
    table image and returns cell structure in pixel coordinates.
    
    The base class provides map_cells_to_pdf() for coordinate conversion.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this TSR engine (e.g., 'tatr', 'surya', 'tableformer')."""
        pass
    
    @abstractmethod
    def predict_cells(self, table_image_crop: np.ndarray) -> List[CellPx]:
        """
        Predict table cell structure from an image crop.
        
        Args:
            table_image_crop: Image of the table as numpy array (H, W, C)
        
        Returns:
            List of CellPx objects with row/col indices and bboxes in pixel coords
        """
        pass
    
    def map_cells_to_pdf(
        self,
        cells_px: List[CellPx],
        crop_bbox_pdf: BBoxPDF,
        image_size: Tuple[int, int],
        crop_offset_px: Tuple[float, float] = (0, 0),
    ) -> List[TableCell]:
        """
        Convert cells from pixel coordinates to PDF coordinates.
        
        The mapping assumes the image was rendered from the PDF region
        defined by crop_bbox_pdf, with the given image size.
        
        Args:
            cells_px: Cells with bboxes in pixel coordinates
            crop_bbox_pdf: The PDF bbox that was rendered to create the image
            image_size: (width, height) of the rendered image in pixels
            crop_offset_px: Optional offset if only part of the image was used
        
        Returns:
            List of TableCell with bbox_pdf in PDF coordinates
        """
        img_width, img_height = image_size
        pdf_width = crop_bbox_pdf[2] - crop_bbox_pdf[0]
        pdf_height = crop_bbox_pdf[3] - crop_bbox_pdf[1]
        
        # Scale factors: pixels -> PDF points
        scale_x = pdf_width / img_width if img_width > 0 else 1.0
        scale_y = pdf_height / img_height if img_height > 0 else 1.0
        
        result = []
        for cell in cells_px:
            # Convert pixel bbox to PDF coordinates
            x0_px, y0_px, x1_px, y1_px = cell.bbox_px
            
            # Apply offset and scale
            x0_pdf = crop_bbox_pdf[0] + (x0_px - crop_offset_px[0]) * scale_x
            y0_pdf = crop_bbox_pdf[1] + (y0_px - crop_offset_px[1]) * scale_y
            x1_pdf = crop_bbox_pdf[0] + (x1_px - crop_offset_px[0]) * scale_x
            y1_pdf = crop_bbox_pdf[1] + (y1_px - crop_offset_px[1]) * scale_y
            
            result.append(TableCell(
                row=cell.row,
                col=cell.col,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                bbox_pdf=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
                text="",  # Text will be filled later from native words
                word_ids=[],
                is_header=cell.is_header,
            ))
        
        return result
    
    def render_crop(
        self,
        page: "fitz.Page",
        bbox_pdf: BBoxPDF,
        dpi: int = 150,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Render a region of a PDF page to an image.
        
        Args:
            page: fitz.Page object
            bbox_pdf: Region to render in PDF coordinates
            dpi: Resolution for rendering (default 150)
        
        Returns:
            Tuple of (image as numpy array, (width, height) in pixels)
        """
        import fitz
        
        # Create clip rectangle
        clip = fitz.Rect(bbox_pdf)
        
        # Calculate zoom factor for desired DPI (PDF is 72 dpi by default)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render
        pix = page.get_pixmap(matrix=mat, clip=clip)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # Convert to RGB if necessary
        if pix.n == 4:  # RGBA
            img = img[:, :, :3]
        
        return img, (pix.width, pix.height)
