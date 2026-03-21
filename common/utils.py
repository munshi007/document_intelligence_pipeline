"""
General utility functions for the pipeline.
"""
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union

logger = logging.getLogger(__name__)

def save_image(image: np.ndarray, filepath: Union[str, Path], description: str = "") -> bool:
    """Save image to file with error handling."""
    try:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if image is None or image.size == 0:
            logger.warning(f"Attempted to save empty image to {filepath}")
            return False

        pil_image = Image.fromarray(image)
        pil_image.save(filepath)
        
        if description:
            logger.debug(f"Saved {description}: {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving image {filepath}: {e}")
        return False
