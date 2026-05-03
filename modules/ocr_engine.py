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
    """Native-First OCR engine using PyMuPDF and fallback VLM hooks."""
    
    def __init__(self):
        """Initialize the OCR engine."""
        self.ocr_type = 'pymupdf_native'
        logger.info("OCREngine: Initialized with PyMuPDF Native-First strategy.")
    
<<<<<<< HEAD
    def is_available(self) -> bool:
        """PyMuPDF is always available if the system is running."""
        return True
=======
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
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
    
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
        [DEPRECATED] Vision-based OCR path. 
        Native extraction via PyMuPDF in TableExtractor/PageProcessor is now prioritized.
        """
        logger.info("OCREngine: Vision-based OCR path invoked (Legacy). Use Specialist VLM for non-native regions.")
        return []
                
    def extract_cell_text(self, cell_image: np.ndarray) -> str:
        """
        Extractor for single table cells. Handed over to Specialist VLM in the Table Coordinator.
        """
        return ""

    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get OCR engine information."""
        return {
            'available': self.is_available(),
            'type': self.ocr_type,
            'dependencies': {"pymupdf": True}
        }