"""
Layout Detection Module - YOLO and Fallback Detection Logic
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, DOCLAYOUT_CONFIG, WEIGHTS_CONFIG

logger = logging.getLogger(__name__)

# Check for optional dependencies
DEPENDENCIES = {
    'doclayout_yolo': False,
    'torch': False,
    'ultralytics': False,
    'layoutparser': False
}

try:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download
    DEPENDENCIES['doclayout_yolo'] = True
except ImportError:
    logger.warning("DocLayout-YOLO not available")

try:
    import torch
    DEPENDENCIES['torch'] = True
except ImportError:
    logger.warning("PyTorch not available")

try:
    from ultralytics import YOLO
    DEPENDENCIES['ultralytics'] = True
except ImportError:
    logger.warning("Ultralytics YOLO not available")

try:
    import layoutparser as lp
    DEPENDENCIES['layoutparser'] = True
except ImportError:
    logger.warning("LayoutParser not available")

class LayoutDetector:
    """Advanced layout detection using DocLayout-YOLO with fallback methods."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the layout detector."""
        self.debug_mode = debug_mode
        self.layout_model = None
        self.layoutparser_model = None
        self.manual_config_path = str(Path(__file__).parent.parent / WEIGHTS_CONFIG['layout_config'])
        self.manual_weights_path = str(Path(__file__).parent.parent / WEIGHTS_CONFIG['layout_weights'])
        self.id2label = DOCLAYOUT_CONFIG['id2label']
        self.layoutparser_label_map = {
            0: "Text",
            1: "Title", 
            2: "List",
            3: "Table",
            4: "Figure"
        }
        # Define preferred labels for voting ties or conflicts
        self.preferred_types = ["Title", "Table", "Figure", "List"]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize layout detection model with proper error handling."""
        try:
            # Initialize DocLayout-YOLO
            if DEPENDENCIES['doclayout_yolo']:
                logger.info("Loading DocLayout-YOLO model...")
                model_file = hf_hub_download(
                    repo_id=DOCLAYOUT_CONFIG['repo_id'],
                    filename=DOCLAYOUT_CONFIG['filename']
                )
                self.layout_model = YOLOv10(model_file)
                self.layout_model.conf_threshold = MODEL_CONFIG['confidence_threshold']
                logger.info("DocLayout-YOLO model loaded successfully")
            elif DEPENDENCIES['ultralytics']:
                logger.info("Loading Ultralytics YOLO as fallback...")
                try:
                    self.layout_model = YOLO('yolov8n.pt')
                    self.id2label = {0: "object"}
                    logger.info("Ultralytics YOLO loaded as fallback")
                except Exception as e:
                    logger.warning(f"Failed to load Ultralytics YOLO: {e}")
                    self.layout_model = None
            else:
                logger.warning("No YOLO model available")
                self.layout_model = None
            
            # Initialize LayoutParser (PubLayNet for semantic refinement)
            if DEPENDENCIES['layoutparser']:
                logger.info("Loading LayoutParser model (Local PubLayNet Assets)...")
                try:
                    # Use local assets to bypass broken Dropbox links
                    if Path(self.manual_config_path).exists() and Path(self.manual_weights_path).exists():
                        self.layoutparser_model = lp.Detectron2LayoutModel(
                            config_path=self.manual_config_path,
                            model_path=self.manual_weights_path,
                            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
                            label_map=self.layoutparser_label_map
                        )
                        logger.info("LayoutParser (PubLayNet) loaded successfully from local assets")
                    else:
                        logger.warning("Local LayoutParser assets missing. Skipping LP initialization.")
                        self.layoutparser_model = None
                except Exception as e:
                    logger.warning(f"LayoutParser initialization failed: {e}")
                    self.layoutparser_model = None
            else:
                logger.info("LayoutParser not available")
                self.layoutparser_model = None
            
            if self.layout_model is None and self.layoutparser_model is None:
                logger.warning("No layout detection models available, using fallback methods")
            
        except Exception as e:
            logger.error(f"Error initializing layout models: {e}")
            self.layout_model = None
            self.layoutparser_model = None
    
    def detect_layout_regions(self, page_image: np.ndarray, debug: bool = True, use_ensemble: bool = True) -> List[Dict[str, Any]]:
        """
        Detect layout regions with proper debugging and validation.
        
        Args:
            page_image: Input page image as numpy array
            debug: Enable debug logging
            use_ensemble: Use ensemble of models if available
            
        Returns:
            List of detected regions with metadata
        """
        logger.info("Starting layout region detection")
        
        # Use ensemble if both models available and requested
        if use_ensemble and self.layout_model is not None and self.layoutparser_model is not None:
            try:
                logger.info("Using ensemble detection")
                all_detections = self.detect_with_ensemble(page_image, debug)
                # Use higher IOU threshold (0.5) to properly deduplicate overlapping tables
                merged_regions = self.merge_detections(all_detections, iou_threshold=0.5, debug=debug)
                logger.info(f"Ensemble detection completed: {len(merged_regions)} regions found")
                return merged_regions
            except Exception as e:
                logger.error(f"Ensemble detection failed: {e}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
                # Fall through to single model detection
        
        # Single model detection (YOLO preferred)
        if self.layout_model is not None:
            try:
                logger.info("Using single model (YOLO) detection")
                detection_image, scale_info = self._prepare_image_for_detection(page_image, debug)
                results = self.layout_model.predict(detection_image, imgsz=DOCLAYOUT_CONFIG['target_size'], verbose=False)
                
                if debug:
                    logger.info(f"Model prediction returned {len(results)} result(s)")
                
                layout_regions = self._process_detection_results(results, scale_info, page_image.shape, debug)
                logger.info(f"Layout detection completed: {len(layout_regions)} regions found")
                return layout_regions
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Try LayoutParser as fallback
        if self.layoutparser_model is not None:
            try:
                logger.info("Using LayoutParser as fallback")
                lp_regions = self._detect_with_layoutparser(page_image, debug)
                logger.info(f"LayoutParser detection completed: {len(lp_regions)} regions found")
                return lp_regions
            except Exception as e:
                logger.error(f"LayoutParser detection failed: {e}")
        
        # Final fallback to CV methods
        logger.warning("No layout models available, using fallback detection")
        return self._fallback_layout_detection(page_image)
    
    def detect_with_ensemble(self, page_image: np.ndarray, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Detect layout regions using ensemble of YOLO and LayoutParser models.
        
        Args:
            page_image: Input page image as numpy array
            debug: Enable debug logging
            
        Returns:
            List of detected regions from ensemble
        """
        logger.info("Starting ensemble layout detection")
        
        all_detections = []
        
        # Run YOLO detection
        if self.layout_model is not None:
            try:
                logger.info("Running YOLO detection...")
                yolo_detections = self._detect_with_yolo(page_image, debug)
                all_detections.extend(yolo_detections)
                logger.info(f"YOLO detected {len(yolo_detections)} regions")
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
        
        # Run LayoutParser detection
        if self.layoutparser_model is not None:
            try:
                logger.info("Running LayoutParser detection...")
                lp_detections = self._detect_with_layoutparser(page_image, debug)
                all_detections.extend(lp_detections)
                logger.info(f"LayoutParser detected {len(lp_detections)} regions")
            except Exception as e:
                logger.error(f"LayoutParser detection failed: {e}")
        
        if not all_detections:
            logger.warning("No detections from ensemble models, using fallback")
            return self._fallback_layout_detection(page_image)
        
        logger.info(f"Ensemble collected {len(all_detections)} total detections")
        return all_detections
    
    def _detect_with_yolo(self, page_image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
        """Run YOLO detection and return regions."""
        detection_image, scale_info = self._prepare_image_for_detection(page_image, debug)
        results = self.layout_model.predict(detection_image, imgsz=DOCLAYOUT_CONFIG['target_size'], verbose=False)
        return self._process_detection_results(results, scale_info, page_image.shape, debug)
    
    def _detect_with_layoutparser(self, page_image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
        """Run LayoutParser detection and return regions."""
        # LayoutParser expects RGB image
        if len(page_image.shape) == 2:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2RGB)
        elif page_image.shape[2] == 4:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2RGB)
        
        # Run detection
        layout = self.layoutparser_model.detect(page_image)
        
        for i, block in enumerate(layout):
            x1, y1, x2, y2 = block.coordinates
            region_type = block.type if hasattr(block, 'type') else 'text'
            confidence = block.score if hasattr(block, 'score') else 0.5
            
            # Standardize label to Title Case
            if region_type.lower() == 'text':
                region_type = 'Text'
            elif 'caption' in region_type.lower():
                region_type = 'Caption'
            else:
                region_type = region_type.title()
                
            region = {
                "type": region_type,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "source": "layout_model", # Standardize source for ensemble merging
                "confidence": float(confidence),
                "region_id": f"lp_{i}",
                "model": "layoutparser"
            }
            regions.append(region)
        
        return regions
    
    def merge_detections(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.3, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Merge overlapping detections using confidence voting and NMS.
        
        Args:
            detections: List of all detections from multiple models
            iou_threshold: IOU threshold for considering regions as overlapping
            debug: Enable debug logging
            
        Returns:
            Merged list of regions
        """
        if not detections:
            return []
        
        # Sort by confidence (descending) and area (descending for tie-breaking)
        detections_sorted = sorted(
            detections,
            key=lambda x: (x['confidence'], self._compute_area(x['bbox'])),
            reverse=True
        )
        
        merged = []
        processed_ids = set()
        
        for detection in detections_sorted:
            det_id = id(detection)
            if det_id in processed_ids:
                continue
            
            # Find overlapping regions
            overlapping = []
            for other in detections_sorted:
                other_id = id(other)
                if other_id == det_id or other_id in processed_ids:
                    continue
                
                iou = self._compute_iou(detection['bbox'], other['bbox'])
                
                # SOTA: Also check for containment (if one is >80% inside another)
                containment_ratio = self._compute_containment(other['bbox'], detection['bbox'])
                
                if iou > iou_threshold or containment_ratio > 0.8:
                    overlapping.append(other)
            
            # If overlapping regions found, use confidence voting with type preference
            if overlapping:
                all_candidates = [detection] + overlapping
                
                # Heuristic: Prefer "Title" or "Table" or "Text" over "Caption" if it's high confidence
                # This fixes "ShenZhenYiKu..." being labeled as caption
                best = self._resolve_label_conflict(all_candidates)
                
                merged_region = {
                    "type": best['type'],
                    "bbox": detection['bbox'],  # Keep highest confidence geometry
                    "source": "ensemble",
                    "confidence": best['confidence'],
                    "region_id": f"ensemble_{len(merged)}",
                    "contributing_models": list(set(d.get('model', d.get('source', 'unknown')) for d in all_candidates))
                }
                
                # Mark all overlapping as processed
                for region in overlapping:
                    processed_ids.add(id(region))
            else:
                # No overlap, keep as is
                merged_region = detection.copy()
                merged_region['region_id'] = f"ensemble_{len(merged)}"
            
            merged.append(merged_region)
            processed_ids.add(det_id)
        
        # Apply NMS to remove remaining duplicates
        final = self._non_max_suppression(merged, iou_threshold)
        
        if debug:
            logger.info(f"Merged {len(detections)} detections into {len(final)} regions")
        
        return final
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Compute union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _compute_area(self, bbox: List[float]) -> float:
        """Compute area of bounding box."""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _compute_containment(self, inner_bbox: List[float], outer_bbox: List[float]) -> float:
        """Compute ratio of inner_bbox area that is inside outer_bbox."""
        x1_min, y1_min, x1_max, y1_max = inner_bbox
        x2_min, y2_min, x2_max, y2_max = outer_bbox
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        inner_area = self._compute_area(inner_bbox)
        
        if inner_area == 0:
            return 0.0
            
        return inter_area / inner_area

    def _resolve_label_conflict(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve label conflicts between overlapping regions from different models."""
        # Primary factor: Confidence
        # Secondary factor: Preferred types (e.g. Title > Caption if high confidence)
        
        if not candidates:
            return {}
            
        # Sort by confidence
        candidates_sorted = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        top = candidates_sorted[0]
        
        # If top is "Caption" but we have a "Title" or "Text" with >70% conf, prefer it
        if top['type'] == 'Caption' and len(candidates_sorted) > 1:
            for cand in candidates_sorted[1:]:
                if cand['type'] in ['Title', 'Text', 'Table'] and cand['confidence'] > 0.7:
                    logger.debug(f"LP Override: Preferring {cand['type']} over Caption for ensemble")
                    return cand
        
        return top
    
    def _non_max_suppression(self, regions: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not regions:
            return []
        
        # Sort by confidence
        sorted_regions = sorted(regions, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, region in enumerate(sorted_regions):
            if i in suppressed:
                continue
            
            keep.append(region)
            
            # Suppress overlapping lower-confidence regions
            for j in range(i + 1, len(sorted_regions)):
                if j in suppressed:
                    continue
                
                iou = self._compute_iou(region['bbox'], sorted_regions[j]['bbox'])
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return keep
    
    def _prepare_image_for_detection(self, page_image: np.ndarray, debug: bool = False) -> tuple:
        """Prepare image for YOLO detection with proper scaling."""
        # Ensure image is in correct format
        if len(page_image.shape) == 3 and page_image.shape[2] == 3:
            pass  # RGB format is correct
        elif len(page_image.shape) == 3 and page_image.shape[2] == 4:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2RGB)
        elif len(page_image.shape) == 2:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2RGB)
        
        # Calculate scaling for target size
        original_height, original_width = page_image.shape[:2]
        target_size = DOCLAYOUT_CONFIG['target_size']
        
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        scale = min(scale_x, scale_y)
        
        # Resize image
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = cv2.resize(page_image, (new_width, new_height))
        
        # Pad to square if needed
        if new_width != target_size or new_height != target_size:
            square_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            square_image[:new_height, :new_width] = resized_image
            detection_image = square_image
        else:
            detection_image = resized_image
        
        scale_info = {
            'scale': scale,
            'original_width': original_width,
            'original_height': original_height,
            'new_width': new_width,
            'new_height': new_height
        }
        
        if debug:
            logger.info(f"Original image size: {original_width}x{original_height}")
            logger.info(f"Detection image size: {detection_image.shape}")
            logger.info(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        return detection_image, scale_info
    
    def _process_detection_results(self, results, scale_info: Dict, original_shape: tuple, debug: bool = False) -> List[Dict[str, Any]]:
        """Process YOLO detection results and convert to region format."""
        layout_regions = []
        
        if not results or len(results) == 0:
            return layout_regions
        
        result = results[0]
        
        if not hasattr(result, 'boxes') or result.boxes is None:
            logger.warning("No boxes attribute in detection results")
            return layout_regions
        
        boxes = result.boxes
        
        if not hasattr(boxes, 'xyxy'):
            logger.warning("No bounding boxes found in detection results")
            return layout_regions
        
        # Extract detection data
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
        classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, 'cpu') else boxes.cls.astype(int)
        
        if debug:
            logger.info(f"Detected {len(xyxy)} regions")
            if len(confs) > 0:
                logger.info(f"Confidence range: {confs.min():.3f} - {confs.max():.3f}")
            else:
                logger.info("No detections with confidence scores")
        
        # Convert detections back to original coordinates
        for bbox, conf, cls in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = bbox
            
            # Scale back to original image coordinates
            x1_orig = x1 / scale_info['scale']
            y1_orig = y1 / scale_info['scale']
            x2_orig = x2 / scale_info['scale']
            y2_orig = y2 / scale_info['scale']
            
            # Get region type and standardize to Title Case (matching PubLayNet mapping)
            region_type = self.id2label.get(cls, f"Unknown_{cls}")
            if region_type.lower() == 'text':
                region_type = 'Text'
            elif 'caption' in region_type.lower():
                region_type = 'Caption'
            else:
                region_type = region_type.title()
            
            # Check if region is completely outside valid area (in padding)
            if (x1_orig >= scale_info['original_width'] or 
                y1_orig >= scale_info['original_height'] or
                x2_orig <= 0 or y2_orig <= 0):
                if debug:
                    logger.warning(f"Rejecting region outside page bounds: {region_type} "
                                 f"bbox=[{x1_orig:.1f}, {y1_orig:.1f}, {x2_orig:.1f}, {y2_orig:.1f}] "
                                 f"page_size=[{scale_info['original_width']}, {scale_info['original_height']}]")
                continue
            
            # Ensure coordinates are within bounds
            x1_orig = max(0, min(x1_orig, scale_info['original_width']))
            y1_orig = max(0, min(y1_orig, scale_info['original_height']))
            x2_orig = max(0, min(x2_orig, scale_info['original_width']))
            y2_orig = max(0, min(y2_orig, scale_info['original_height']))
            
            # Check for valid region after clipping
            if x2_orig <= x1_orig or y2_orig <= y1_orig:
                if debug:
                    logger.warning(f"Rejecting invalid region after clipping: {region_type}")
                continue
            
            region = {
                "type": region_type,
                "bbox": [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                "source": "layout_model",
                "confidence": float(conf),
                "region_id": f"layout_{len(layout_regions)}",
                "model_class": int(cls)
            }
            layout_regions.append(region)
        
        if debug:
            logger.info(f"Final layout regions detected: {len(layout_regions)}")
            for region in layout_regions:
                logger.info(f"  {region['type']}: confidence={region['confidence']:.3f}")
        
        return layout_regions
    
    def _fallback_layout_detection(self, page_image: np.ndarray) -> List[Dict[str, Any]]:
        """Enhanced fallback layout detection using computer vision."""
        logger.info("Using fallback layout detection")
        regions = []
        
        try:
            gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
            height, width = page_image.shape[:2]
            
            # 1. Detect horizontal lines (potential table borders)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # 2. Detect vertical lines (potential table borders)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # 3. Combine lines to find table regions
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # 4. Find contours for potential tables
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size - tables should be reasonably large
                if area > (width * height * 0.01) and w > 100 and h > 50:
                    region = {
                        "type": "Table",
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "source": "fallback_cv",
                        "confidence": 0.7,
                        "region_id": f"fallback_table_{i}",
                        "detection_method": "line_detection"
                    }
                    regions.append(region)
            
            # 5. Detect text blocks using connected components
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Remove lines to focus on text
            no_lines = cv2.subtract(thresh, table_mask)
            
            # Find text contours
            text_contours, _ = cv2.findContours(no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(text_contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter text regions
                if area > 100 and w > 20 and h > 10:
                    # Classify based on aspect ratio and size
                    aspect_ratio = w / h
                    
                    if h > 20 and aspect_ratio < 10:  # Likely heading or title
                        region_type = "Title"
                    elif aspect_ratio > 5:  # Wide text, likely paragraph
                        region_type = "Text"
                    else:
                        region_type = "Text"
                    
                    region = {
                        "type": region_type,
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "source": "fallback_cv",
                        "confidence": 0.6,
                        "region_id": f"fallback_text_{i}",
                        "detection_method": "contour_analysis"
                    }
                    regions.append(region)
            
            logger.info(f"Fallback detection found {len(regions)} regions")
            return regions
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if layout detection model is available."""
        return self.layout_model is not None
    
    def get_dependencies(self) -> Dict[str, bool]:
        """Get dependency status information."""
        return DEPENDENCIES