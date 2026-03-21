"""
SOTA Distillation Agent.
Captures Teacher (GPT-4o/InternVL2-76B) responses to create a high-quality 
dataset for training smaller Student models.
"""

import json
import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class DistillationAgent:
    """Agent for automated Teacher-Student dataset generation."""
    
    def __init__(self, dataset_dir: str = "research/dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_file = self.dataset_dir / "dataset.jsonl"
        
        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def capture(self, image: Image.Image, prompt: str, response: Any, metadata: Optional[Dict] = None):
        """
        Captures a VLM interaction and saves it to the dataset.
        
        Args:
            image: The PIL Image used for the VLM call.
            prompt: The text prompt sent to the VLM.
            response: The structured response (Pydantic model or dict) from the VLM.
            metadata: Optional details (page_num, model_name, etc.)
        """
        try:
            # SOTA: Deduplication Strategy
            # Use metadata to construct a stable filename for the page
            # If the user provides a 'image_id' in metadata, we trust it over the hash.
            image_id = metadata.get('image_id') if metadata else None
            
            if image_id:
                # Sanitize image_id for filename
                image_filename = f"{image_id.replace('/', '_').replace(' ', '_')}.jpg"
                image_path = self.images_dir / image_filename
            else:
                # Fallback to hash-based dedup for ad-hoc crops
                img_byte_arr = image.tobytes()
                img_hash = hashlib.sha256(img_byte_arr).hexdigest()
                image_filename = f"crop_{img_hash[:16]}.jpg"
                image_path = self.images_dir / image_filename

            # 1. Save the Image (if not already exists)
            if not image_path.exists():
                # Ensure it's RGB for JPEG
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(image_path, format="JPEG", quality=90)
                logger.info(f"DistillationAgent: Saved NEW image {image_filename}")
            else:
                logger.debug(f"DistillationAgent: Reusing existing image {image_filename}")
            
            # 2. Process Response (ensure it's a dict)
            if hasattr(response, "dict"):
                response_data = response.dict()
            else:
                response_data = response
                
            # 3. Create Dataset Entry (ShareGPT / HuggingFace format)
            timestamp = int(time.time() * 1000)
            entry_id = f"capture_{timestamp}"
            
            entry = {
                "id": entry_id,
                "metadata": metadata or {},
                "image": f"research/dataset/images/{image_filename}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": json.dumps(response_data, indent=2)}
                ]
            }
            
            # 4. Append to JSONL
            with open(self.labels_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                
            logger.info(f"DistillationAgent: Captured entry {entry_id} (Image: {image_filename})")
            
        except Exception as e:
            logger.error(f"DistillationAgent: Failed to capture interaction: {e}")
