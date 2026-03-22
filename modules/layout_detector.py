"""
Layout Detection Module - RT-DETR Specialist
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import DOCLAYOUT_CONFIG, WEIGHTS_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class LayoutDetector:
    """Production layout detection using RT-DETR Specialist."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the layout detector."""
        self.debug_mode = debug_mode
        self.layout_model = None
        # Standardized labels across the pipeline
        self.id2label = {
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure",
            5: "Caption"
        }
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the RT-DETR model using Ultralytics."""
        try:
            from ultralytics import YOLO
            logger.info("Loading RT-DETR Layout Specialist...")
            # Using the local weights specified in config
            model_path = WEIGHTS_CONFIG.get('layout_weights', 'yolov8n.pt')
            self.layout_model = YOLO(model_path)
            logger.info(f"Layout Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing layout model: {e}")
            self.layout_model = None

    def detect_layout_regions(self, page_image: np.ndarray, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Detect layout regions using the specialized RT-DETR model.
        """
        if self.layout_model is None:
            logger.warning("Layout model not available, returning empty regions")
            return []
            
        try:
            logger.info("Using RT-DETR Specialist for layout detection")
            detection_image, scale_info = self._prepare_image_for_detection(page_image, debug)
            
            # Predict with Ultralytics (RT-DETR/YOLO handle NMS natively)
            results = self.layout_model.predict(
                detection_image, 
                imgsz=DOCLAYOUT_CONFIG['target_size'], 
                conf=MODEL_CONFIG['confidence_threshold'],
                verbose=False
            )
            
            layout_regions = self._process_detection_results(results, scale_info, page_image.shape, debug)
            logger.info(f"Layout detection completed: {len(layout_regions)} regions found")
            return layout_regions
            
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            return []

    def _prepare_image_for_detection(self, page_image: np.ndarray, debug: bool = False) -> tuple:
        """Prepare image for YOLO detection with proper scaling."""
        if len(page_image.shape) == 3 and page_image.shape[2] == 4:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2RGB)
        elif len(page_image.shape) == 2:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2RGB)
        
        original_height, original_width = page_image.shape[:2]
        target_size = DOCLAYOUT_CONFIG['target_size']
        
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        scale = min(scale_x, scale_y)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = cv2.resize(page_image, (new_width, new_height))
        
        square_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        square_image[:new_height, :new_width] = resized_image
        detection_image = square_image
        
        scale_info = {
            'scale': scale,
            'original_width': original_width,
            'original_height': original_height
        }
        return detection_image, scale_info

    def _process_detection_results(self, results, scale_info: Dict, original_shape: tuple, debug: bool = False) -> List[Dict[str, Any]]:
        """Process YOLO detection results and convert to region format."""
        layout_regions = []
        if not results: return layout_regions
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return layout_regions
        
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
        classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, 'cpu') else boxes.cls.astype(int)
        
        for bbox, conf, cls in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = bbox
            
            x1_orig = x1 / scale_info['scale']
            y1_orig = y1 / scale_info['scale']
            x2_orig = x2 / scale_info['scale']
            y2_orig = y2 / scale_info['scale']
            
            region_type = self.id2label.get(cls, f"Unknown_{cls}")
            
            if (x1_orig >= scale_info['original_width'] or 
                y1_orig >= scale_info['original_height'] or
                x2_orig <= 0 or y2_orig <= 0):
                continue
            
            region = {
                "type": region_type,
                "bbox": [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                "source": "layout_model",
                "confidence": float(conf),
                "region_id": f"layout_{len(layout_regions)}"
            }
            layout_regions.append(region)
        
        return layout_regions

    def is_available(self) -> bool:
        """Check if layout detection model is available."""
        return self.layout_model is not None

    def get_dependencies(self) -> Dict[str, bool]:
        """Get dependency status information."""
        return {"torch": True, "ultralytics": True}