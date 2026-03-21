"""
TATR (Table Transformer) Engine adapter for Tables v2.

Wraps the Microsoft Table Transformer model as a TSREngine.
"""

import logging
from typing import List, Optional
import numpy as np

from .base import TSREngine, CellPx

logger = logging.getLogger(__name__)


class TATREngine(TSREngine):
    """
    TSR Engine wrapping Microsoft's Table Transformer (TATR).
    
    Uses the pre-loaded model from TableExtractor to avoid
    duplicate model loading.
    """
    
    def __init__(
        self,
        model,
        processor,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model: TableTransformerForObjectDetection instance
            processor: AutoImageProcessor instance
            confidence_threshold: Minimum confidence for predictions
        """
        self.model = model
        self.processor = processor
        self.confidence_threshold = confidence_threshold
        
        # TATR label maps
        self.structure_id2label = {
            0: "table",
            1: "table column",
            2: "table row", 
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell",
        }
    
    @property
    def name(self) -> str:
        return "tatr"
    
    def predict_cells(self, table_image_crop: np.ndarray) -> List[CellPx]:
        """
        Predict table cell structure using TATR.
        
        Returns cells with row/col indices derived from detected
        rows, columns, and spanning cells.
        """
        try:
            import torch
            from PIL import Image
        except ImportError:
            logger.warning("torch/PIL not available for TATR")
            return []
        
        try:
            # Convert to PIL
            if len(table_image_crop.shape) == 2:
                pil_image = Image.fromarray(table_image_crop).convert("RGB")
            else:
                pil_image = Image.fromarray(table_image_crop)
            
            width, height = pil_image.size
            
            # Process image
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
                size={"height": 800, "width": 800}
            )
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([[height, width]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes
            )[0]
            
            # Parse TATR output into cells
            return self._parse_tatr_results(results, width, height)
            
        except Exception as e:
            logger.warning(f"TATR prediction failed: {e}")
            return []
    
    def _parse_tatr_results(
        self,
        results: dict,
        img_width: int,
        img_height: int,
    ) -> List[CellPx]:
        """
        Parse TATR detection results into CellPx objects.
        
        TATR outputs rows, columns, and headers. We derive cells
        from row/column intersections.
        """
        boxes = results["boxes"].tolist()
        labels = results["labels"].tolist()
        scores = results["scores"].tolist()
        
        # Separate by type
        rows = []
        cols = []
        headers = []
        spanning_cells = []
        
        for box, label, score in zip(boxes, labels, scores):
            label_name = self.structure_id2label.get(label, "unknown")
            
            if label_name == "table row":
                rows.append({"bbox": box, "score": score})
            elif label_name == "table column":
                cols.append({"bbox": box, "score": score})
            elif label_name == "table column header":
                headers.append({"bbox": box, "score": score})
            elif label_name == "table spanning cell":
                spanning_cells.append({"bbox": box, "score": score})
        
        # Sort rows by Y, columns by X
        rows.sort(key=lambda r: r["bbox"][1])
        cols.sort(key=lambda c: c["bbox"][0])
        
        if not rows or not cols:
            logger.debug("TATR: No rows or columns detected")
            return []
        
        # Create cells from row/column intersections
        cells = []
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                # Cell bbox = intersection
                x0 = max(row["bbox"][0], col["bbox"][0])
                y0 = max(row["bbox"][1], col["bbox"][1])
                x1 = min(row["bbox"][2], col["bbox"][2])
                y1 = min(row["bbox"][3], col["bbox"][3])
                
                # Valid intersection?
                if x1 > x0 and y1 > y0:
                    is_header = row_idx == 0 or self._overlaps_header(
                        (x0, y0, x1, y1), headers
                    )
                    
                    cells.append(CellPx(
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        bbox_px=(x0, y0, x1, y1),
                        confidence=min(row["score"], col["score"]),
                        is_header=is_header,
                    ))
        
        logger.debug(f"TATR: Generated {len(cells)} cells from {len(rows)} rows, {len(cols)} cols")
        return cells
    
    def _overlaps_header(
        self,
        cell_bbox: tuple,
        headers: List[dict],
    ) -> bool:
        """Check if cell overlaps with any header region."""
        x0, y0, x1, y1 = cell_bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        
        for header in headers:
            hx0, hy0, hx1, hy1 = header["bbox"]
            if hx0 <= cx <= hx1 and hy0 <= cy <= hy1:
                return True
        
        return False
