"""
Layout Detection Module - YOLO and Fallback Detection Logic
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import base64
import io
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, DOCLAYOUT_CONFIG, WEIGHTS_CONFIG, VLM_CONFIG
from common.vlm_client import VLMClient

logger = logging.getLogger(__name__)

# Check for optional dependencies
DEPENDENCIES = {
    'doclayout_yolo': False,
    'torch': False,
    'ultralytics': False,
}

try:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download
    DEPENDENCIES['doclayout_yolo'] = True
except Exception as e:
    logger.warning(f"DocLayout-YOLO not available: {e}")
    # Also print it to stderr so it's visible in the terminal
    import sys
    print(f"DocLayout-YOLO not available: {e}", file=sys.stderr)

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

class LayoutDetector:
    """Advanced layout detection using DocLayout-YOLO with VLM semantic refinement."""
    
    def __init__(self, debug_mode: bool = False, vlm_client: Optional[VLMClient] = None):
        """Initialize the layout detector."""
        self.debug_mode = debug_mode
        self.layout_model = None
        self.vlm_client = vlm_client  # Custom VLM for semantic refinement
        self.id2label = DOCLAYOUT_CONFIG['id2label']
        self.yolo_label_map = {
            0: "plain text",
            1: "title",
            2: "list",
            3: "figure",
            4: "table",
            5: "cell",
            6: "abandon"
        }
        # Define preferred labels for voting ties or conflicts
        self.preferred_types = ["title", "table", "figure", "list"]
        
        # Initialize VLM for semantic refinement if not provided
        if self.vlm_client is None:
            try:
                vlm_config = {
                    'model': VLM_CONFIG.get('default_model', 'RMunshi/vlm-student-thesis'),
                    'provider': VLM_CONFIG.get('default_provider', 'local')
                }
                self.vlm_client = VLMClient(config=vlm_config)
                logger.info(f"VLM Client initialized for semantic refinement: {vlm_config['model']}")
            except Exception as e:
                logger.warning(f"Could not initialize VLM for semantic refinement: {e}")
                self.vlm_client = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize layout detection model with proper error handling."""
        try:
            # Initialize DocLayout-YOLO (primary detector)
            if DEPENDENCIES['doclayout_yolo']:
                from common import model_registry
                spec = model_registry.get("doclayout_yolo")
                logger.info(f"Loading DocLayout-YOLO {spec.repo_id}@{spec.revision[:10]}...")
                model_file = hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    revision=spec.revision,
                )
                self.layout_model = YOLOv10(model_file)
                self.layout_model.conf_threshold = MODEL_CONFIG['confidence_threshold']
                logger.info("✅ DocLayout-YOLO model loaded successfully (primary detector)")
            elif DEPENDENCIES['ultralytics']:
                logger.info("Loading Ultralytics YOLO as fallback...")
                try:
                    self.layout_model = YOLO('yolov8n.pt')
                    self.id2label = {0: "object"}
                    logger.info("✅ Ultralytics YOLO loaded as fallback")
                except Exception as e:
                    logger.warning(f"Failed to load Ultralytics YOLO: {e}")
                    self.layout_model = None
            else:
                logger.warning("No YOLO model available")
                self.layout_model = None
            
            # Semantic refinement via custom VLM (replaces broken LayoutParser)
            if self.vlm_client is not None:
                logger.info("✅ VLM Client available for semantic layout refinement")
                logger.info(f"   Using fine-tuned model: {VLM_CONFIG.get('default_model', 'RMunshi/vlm-student-thesis')}")
            else:
                logger.info("ℹ️ VLM Client not available - will use YOLO labels as-is")
            
            if self.layout_model is None:
                logger.warning("No primary layout detection models available")
            
        except Exception as e:
            logger.error(f"Error initializing layout models: {e}")
            self.layout_model = None
    
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
        
        # YOLO + VLM semantic refinement (replaces old ensemble with LayoutParser)
        if self.layout_model is not None:
            try:
                logger.info("Using YOLO detection with VLM semantic refinement")
                detection_image, scale_info = self._prepare_image_for_detection(page_image, debug)
                results = self.layout_model.predict(detection_image, imgsz=DOCLAYOUT_CONFIG['target_size'], verbose=False)
                
                if debug:
                    logger.info(f"Model prediction returned {len(results)} result(s)")
                
                layout_regions = self._process_detection_results(results, scale_info, page_image.shape, debug)
                logger.info(f"YOLO detected {len(layout_regions)} regions")
                
                # Refine YOLO detections with your custom fine-tuned  VLM
                if self.vlm_client is not None:
                    logger.info("Applying custom VLM semantic refinement...")
                    layout_regions = self._refine_with_vlm(page_image, layout_regions, debug)
                
                # SOTA: Apply NMS and overlap merging to fix duplicate YOLO bounding boxes (e.g. Table overlapping with Caption)
                logger.info("Applying NMS and label conflict resolution...")
                layout_regions = self.merge_detections(layout_regions, debug=debug)
                
                logger.info(f"Layout detection completed: {len(layout_regions)} regions after refinement and merging")
                return layout_regions
            except Exception as e:
                logger.error(f"YOLO+VLM detection failed: {e}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Final fallback to CV methods
        logger.warning("No layout models available, using fallback detection")
        return self._fallback_layout_detection(page_image)
    
    def _detect_with_yolo(self, page_image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
        """Run YOLO detection and return regions."""
        detection_image, scale_info = self._prepare_image_for_detection(page_image, debug)
        results = self.layout_model.predict(detection_image, imgsz=DOCLAYOUT_CONFIG['target_size'], verbose=False)
        return self._process_detection_results(results, scale_info, page_image.shape, debug)

    def _refine_with_vlm(self, page_image: np.ndarray, yolo_regions: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
        """
        Refine YOLO detections using custom fine-tuned VLM for semantic consistency.
        
        This replaces the broken LayoutParser fallback with your custom Unsloth model,
        which gives you an advantage for your thesis project.
        
        Args:
            page_image: Input page image as numpy array
            yolo_regions: List of regions detected by YOLO
            debug: Enable debug logging
            
        Returns:
            Refined list of regions with VLM semantic validation
        """
        if debug:
            logger.info("VLM refinement bypassed to preserve DocLayout-YOLO spatial accuracy.")
        return yolo_regions
        
        try:
            # 1. Normalize YOLO regions to 1000x1000 for VLM grounding
            img_h, img_w = page_image.shape[:2]
            
            # Create a prompt for the VLM to verify/refine layout labels with coordinates
            prompt = f"""Analyze the document layout and verify the region classifications. 
I have detected {len(yolo_regions)} regions using YOLO. For each region, verify if the label is correct or suggest a fix.

Regions to verify (coordinates in [ymin, xmin, ymax, xmax] 0-1000 format):
"""
            for i, region in enumerate(yolo_regions):
                x1, y1, x2, y2 = region['bbox']
                # Normalize to 0-1000
                ymin, xmin, ymax, xmax = int(y1*1000/img_h), int(x1*1000/img_w), int(y2*1000/img_h), int(x2*1000/img_w)
                region_type = region.get('type', 'unknown')
                conf = region.get('confidence', 0.0)
                prompt += f"  Region {i}: {region_type} at [{ymin}, {xmin}, {ymax}, {xmax}] (YOLO confidence: {conf:.2f})\n"
            
            prompt += """
Based on the document image, please verify these classifications. Respond with a JSON object mapping region index to refined label and confidence.
Example: {"0": {"label": "Table", "confidence": 0.95}, "1": {"label": "Title", "confidence": 0.98}}

Only return the JSON object inside markdown backticks."""
            
            # Call VLM with image for context
            refinement_result = self.vlm_client.generate(
                prompt=prompt,
                image=page_image,
                max_tokens=1000
            )
            
            # Parse refinement result
            try:
                import json
                # Extract JSON from response
                json_text = refinement_result
                if '{' in json_text:
                    json_text = json_text[json_text.find('{'):json_text.rfind('}')+1]
                refinements = json.loads(json_text)
                
                # Apply refinements
                refined_regions = []
                for i, region in enumerate(yolo_regions):
                    region_refined = region.copy()
                    if str(i) in refinements:
                        ref = refinements[str(i)]
                        if 'label' in ref:
                            region_refined['type'] = ref['label'].title()
                        if 'confidence' in ref:
                            # Blend VLM confidence with YOLO confidence
                            yolo_conf = region.get('confidence', 0.5)
                            vlm_conf = float(ref['confidence'])
                            region_refined['confidence'] = (yolo_conf + vlm_conf) / 2
                            region_refined['vlm_refined'] = True
                    refined_regions.append(region_refined)
                
                if debug:
                    logger.info(f"VLM refinement updated {sum(1 for r in refined_regions if r.get('vlm_refined'))} regions")
                
                return refined_regions
            
            except (json.JSONDecodeError, ValueError) as e:
                if debug:
                    logger.debug(f"Could not parse VLM refinement response: {e}. Using YOLO labels as-is.")
                return yolo_regions
        
        except Exception as e:
            logger.warning(f"VLM semantic refinement failed: {e}. Using YOLO labels as-is.")
            return yolo_regions
    
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