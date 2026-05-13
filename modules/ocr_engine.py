"""
OCR Engine Module - PaddleOCR Integration and Wrappers
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import OCR_CONFIG

logger = logging.getLogger(__name__)

# Check for OCR dependencies
DEPENDENCIES = {
    'paddleocr': False,
    'torch': False
}

try:
    from paddleocr import PaddleOCR
    DEPENDENCIES['paddleocr'] = True
except ImportError:
    logger.warning("PaddleOCR not available")

try:
    import torch
    DEPENDENCIES['torch'] = True
except ImportError:
    logger.warning("PyTorch not available")

class OCREngine:
    """OCR engine with PaddleOCR integration and fallback methods."""
    
    def __init__(self):
        """Initialize the OCR engine."""
        self.ocr_reader = None
        self.ocr_type = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR reader."""
        try:
            if DEPENDENCIES['paddleocr']:
                logger.info("Initializing PaddleOCR (GPU-Accelerated)...")
                use_gpu = False
                if DEPENDENCIES['torch']:
                    import torch
                    use_gpu = torch.cuda.is_available()
                
                self.ocr_reader = PaddleOCR(
                    use_angle_cls=OCR_CONFIG['use_angle_cls'],
                    lang=OCR_CONFIG['lang'],
                    use_gpu=use_gpu,
                    ir_optim=True,
                    show_log=False
                )
                self.ocr_type = 'paddle'
                logger.info(f"PaddleOCR initialized successfully (use_gpu={use_gpu})")
                return
            
            logger.warning("PaddleOCR not available")
            self.ocr_reader = None
            self.ocr_type = None
            
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR: {e}")
            self.ocr_reader = None
            self.ocr_type = None
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.ocr_reader is not None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        Applies denoising and contrast enhancement.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            import cv2
            
            # Skip if image is too small
            if image.shape[0] < 10 or image.shape[1] < 10:
                return image
            
            # Convert to BGR if needed (OpenCV expects BGR)
            if len(image.shape) == 2:
                # Grayscale
                gray = image
            elif image.shape[2] == 4:
                # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                return image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Light denoising (preserve text edges)
            denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Convert back to 3-channel for PaddleOCR
            result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
            
            logger.debug("Image preprocessing applied: CLAHE + denoising")
            return result
            
        except Exception as e:
            logger.debug(f"Image preprocessing failed, using original: {e}")
            return image
    
    def extract_text_from_image(self, image: np.ndarray, preprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image as numpy array
            preprocess: Whether to apply preprocessing (contrast/denoise)
            
        Returns:
            List of text boxes with bounding boxes and confidence scores
        """
        if not self.is_available():
            logger.warning("OCR not available")
            return []
        
        try:
            # Apply preprocessing for better OCR accuracy
            if preprocess:
                image = self._preprocess_image(image)
            
            if self.ocr_type == 'paddle':
                return self._extract_with_paddleocr(image)
            else:
                logger.warning("No OCR method available")
                return []
                
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            return []
    
    def _extract_with_paddleocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using PaddleOCR with font metadata."""
        try:
            # PaddleOCR 3.x returns OCRResult objects
            ocr_results = self.ocr_reader.ocr(image)
            text_boxes = []
            
            if not ocr_results or len(ocr_results) == 0:
                return []
            
            # Get the first result (for single image)
            result = ocr_results[0]
            
            # New API: result is a dict-like OCRResult object (PP-Structure)
            if hasattr(result, 'keys'):
                dt_polys = result.get('dt_polys', [])
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                
                # Process each detected text region
                for idx, (poly, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
                    # Convert polygon to bbox [x1, y1, x2, y2]
                    x_coords = [point[0] for point in poly]
                    y_coords = [point[1] for point in poly]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    # Estimate font size from bbox height
                    font_size = y2 - y1
                    
                    # Detect bold text heuristically
                    font_weight = self._estimate_font_weight(image, [x1, y1, x2, y2])
                    
                    text_boxes.append({
                        'text': text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'font_size': float(font_size),
                        'font_family': 'unknown',
                        'font_weight': font_weight,
                        'font_style': 'normal',
                        'line_id': idx,
                        'paragraph_id': -1
                    })
            
            # Standard PP-OCR list output: [[[[x1,y1],...], ("text", score)], ...]
            elif isinstance(result, list):
                for idx, line in enumerate(result):
                    if len(line) < 2:
                        continue
                        
                    box = line[0] # List of points
                    text_info = line[1] # (text, score)
                    
                    if not text_info or len(text_info) < 2:
                        continue
                        
                    text = text_info[0]
                    score = text_info[1]
                    
                    # Convert box points to bbox [x1, y1, x2, y2]
                    # box is usually [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    font_size = y2 - y1
                    font_weight = self._estimate_font_weight(image, [x1, y1, x2, y2])
                    
                    text_boxes.append({
                        'text': text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'font_size': float(font_size),
                        'font_family': 'unknown',
                        'font_weight': font_weight,
                        'font_style': 'normal',
                        'line_id': idx,
                        'paragraph_id': -1
                    })
            
            logger.info(f"PaddleOCR extracted {len(text_boxes)} text boxes with metadata")
            return text_boxes
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return []
    
    def _estimate_font_weight(self, image: np.ndarray, bbox: List[float]) -> str:
        """
        Estimate font weight (normal/bold) from text region.
        Uses pixel density as a heuristic.
        """
        try:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return 'normal'
            
            # Extract text region
            text_region = image[y1:y2, x1:x2]
            
            if text_region.size == 0:
                return 'normal'
            
            # Convert to grayscale if needed
            if len(text_region.shape) == 3:
                import cv2
                gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            else:
                gray = text_region
            
            # Calculate pixel density (ratio of dark pixels)
            # Bold text typically has higher density
            threshold = 128
            dark_pixels = np.sum(gray < threshold)
            total_pixels = gray.size
            density = dark_pixels / total_pixels if total_pixels > 0 else 0
            
            # Heuristic threshold for bold detection
            return 'bold' if density > 0.3 else 'normal'
            
        except Exception as e:
            logger.debug(f"Failed to estimate font weight: {e}")
            return 'normal'
    
    def extract_cell_text(self, cell_image: np.ndarray) -> str:
        """
        Extract text from a single table cell.
        
        Args:
            cell_image: Cell image as numpy array
            
        Returns:
            Extracted text content
        """
        if cell_image.size == 0 or not self.is_available():
            return ""
        
        try:
            if self.ocr_type == 'paddle':
                ocr_result = self.ocr_reader.ocr(cell_image, cls=True)
                if ocr_result and ocr_result[0]:
                    return " ".join([line[1][0] for line in ocr_result[0]])
            else:
                return ""
                
        except Exception as e:
            logger.debug(f"Cell OCR failed: {e}")
        
        return ""
    
    def detect_headings(
        self, 
        ocr_results: List[Dict[str, Any]], 
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect headings using adaptive font statistics from DocumentProfile.
        
        Args:
            ocr_results: List of OCR results with font metadata
            doc_profile: DocumentProfile with adaptive thresholds
            
        Returns:
            List of heading regions
        """
        if not ocr_results:
            return []
        
        headings = []
        
        # Get adaptive threshold from document profile
        if doc_profile and hasattr(doc_profile, 'thresholds'):
            font_threshold = doc_profile.thresholds.heading_font_size
        else:
            # Fallback: compute threshold from current results
            font_sizes = [r.get('font_size', 12) for r in ocr_results if 'font_size' in r]
            if font_sizes:
                mean_size = np.mean(font_sizes)
                std_size = np.std(font_sizes)
                font_threshold = mean_size + 1.5 * std_size
            else:
                font_threshold = 18.0  # Default
        
        logger.info(f"Using adaptive heading threshold: {font_threshold:.2f}")
        
        # Detect headings based on font size and weight
        for result in ocr_results:
            font_size = result.get('font_size', 0)
            font_weight = result.get('font_weight', 'normal')
            
            # Heading if: large font OR bold text
            if font_size > font_threshold or font_weight == 'bold':
                heading = result.copy()
                heading['type'] = 'heading'
                heading['is_heading'] = True
                headings.append(heading)
        
        logger.info(f"Detected {len(headings)} headings from {len(ocr_results)} OCR results")
        return headings
    
    def filter_low_confidence(
        self,
        ocr_results: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter OCR results by confidence score using adaptive threshold.
        
        Args:
            ocr_results: List of OCR results with confidence scores
            doc_profile: DocumentProfile with adaptive thresholds
            min_confidence: Optional manual confidence threshold (overrides adaptive)
            
        Returns:
            Filtered list of OCR results
        """
        if not ocr_results:
            return []
        
        # Determine confidence threshold
        if min_confidence is not None:
            threshold = min_confidence
        elif doc_profile and hasattr(doc_profile, 'thresholds'):
            threshold = doc_profile.thresholds.confidence_threshold
        else:
            # Fallback: compute adaptive threshold (25th percentile)
            confidences = [r.get('confidence', 0) for r in ocr_results]
            if confidences:
                threshold = float(np.percentile(confidences, 25))
            else:
                threshold = 0.5  # Default
        
        logger.info(f"Using confidence threshold: {threshold:.3f}")
        
        # Filter results
        filtered = [r for r in ocr_results if r.get('confidence', 0) >= threshold]
        
        logger.info(f"Filtered {len(ocr_results)} results to {len(filtered)} (removed {len(ocr_results) - len(filtered)} low-confidence)")
        return filtered
    
    def weight_text_importance(
        self,
        ocr_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Weight text importance by confidence scores.
        Adds 'importance' field based on confidence.
        
        Args:
            ocr_results: List of OCR results with confidence scores
            
        Returns:
            OCR results with importance weights
        """
        if not ocr_results:
            return []
        
        # Normalize confidence scores to importance weights (0-1)
        confidences = [r.get('confidence', 0.5) for r in ocr_results]
        
        if confidences:
            min_conf = min(confidences)
            max_conf = max(confidences)
            conf_range = max_conf - min_conf if max_conf > min_conf else 1.0
            
            for result in ocr_results:
                conf = result.get('confidence', 0.5)
                # Normalize to 0-1 range
                importance = (conf - min_conf) / conf_range if conf_range > 0 else 0.5
                result['importance'] = float(importance)
        else:
            # Default importance
            for result in ocr_results:
                result['importance'] = 0.5
        
        return ocr_results
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get OCR engine information."""
        return {
            'available': self.is_available(),
            'type': self.ocr_type,
            'dependencies': DEPENDENCIES
        }