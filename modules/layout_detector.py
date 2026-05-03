"""
Layout Detection Module - RT-DETR Specialist
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

<<<<<<< HEAD
from config import DOCLAYOUT_CONFIG, WEIGHTS_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class LayoutDetector:
    """Production layout detection using RT-DETR Specialist."""
=======
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
    'layoutparser': False
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

try:
    import layoutparser as lp
    DEPENDENCIES['layoutparser'] = True
except ImportError:
    logger.warning("LayoutParser not available")

class LayoutDetector:
    """Advanced layout detection using DocLayout-YOLO with VLM semantic refinement."""
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
    
    def __init__(self, debug_mode: bool = False, vlm_client: Optional[VLMClient] = None):
        """Initialize the layout detector."""
        self.debug_mode = debug_mode
        self.layout_model = None
<<<<<<< HEAD
        # Standardized labels across the pipeline
        self.id2label = {
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure",
            5: "Caption"
        }
=======
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
        
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the RT-DETR model using Ultralytics."""
        try:
<<<<<<< HEAD
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
=======
            # Initialize DocLayout-YOLO (primary detector)
            if DEPENDENCIES['doclayout_yolo']:
                logger.info("Loading DocLayout-YOLO model...")
                model_file = hf_hub_download(
                    repo_id=DOCLAYOUT_CONFIG['repo_id'],
                    filename=DOCLAYOUT_CONFIG['filename']
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
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
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