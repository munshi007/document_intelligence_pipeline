"""
Table Structure Model Module
Encapsulates the Table Transformer (TATR) model and processor.
"""

import logging
from typing import Any, Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Table Transformer (TATR) support
TATR_AVAILABLE = False
try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TATR_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TableStructureModel:
    """
    Wrapper for Microsoft's Table Transformer (TATR) structure recognition model.
    Handles loading and providing access to the model and processor.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.available = False
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        self._load_model()

    def _load_model(self):
        """Load the TATR model and processor."""
        if not (TATR_AVAILABLE and TORCH_AVAILABLE):
            logger.warning("TableStructureModel: Dependencies (transformers/torch) not met.")
            return

        try:
            logger.info("Loading Table Transformer (TATR) structure model...")
            self.processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition-v1.1-all"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition-v1.1-all"
            )
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            logger.info(f"Table Transformer (TATR) loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load Table Transformer: {e}")
            self.available = False
            self.model = None
            self.processor = None

    def get_components(self) -> Tuple[Any, Any]:
        """
        Get the raw model and processor.
        Returns:
            (model, processor) tuple, or (None, None) if not available.
        """
        return self.model, self.processor

    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self.available
