"""
Distillation Agent: Librarian Data Capture
===========================================
Captures 'Gold Standard' extraction pairs (Prompt + Thoughts + JSON)
from the Teacher (GPT-4o) to train local Student models (Qwen2.5/Llama).
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DistillationAgent:
    """
    An observer/collector that records high-fidelity extraction interactions.
    Saves data in JSONL format for easy ingestion by Unsloth/SFT.
    """
    
    def __init__(self, dataset_dir: str = "research/distilled_labels"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # We use a daily JSONL file to prevent massive single files
        self.log_path = self.dataset_dir / f"librarian_sft_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        logger.info(f"DistillationAgent active. Saving to {self.log_path}")
        
    def capture(
        self, 
        image: Optional[Any], 
        prompt: str, 
        result: BaseModel, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Records a single extraction event.
        Logic: Only saves if reasoning_thoughts are present and meaningful.
        """
        try:
            # 1. Extract metadata and result
            res_dict = result.model_dump()
            thoughts = res_dict.get("reasoning_thoughts", "")
            
            # 2. Skip trivial or failed extractions
            if not thoughts or len(thoughts) < 10:
                logger.debug("Skipping distillation: No significant reasoning captured.")
                return

            # 3. Create instruction-tuning record
            record = {
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "instruction": prompt,
                "input_context": "", # This is usually part of the prompt in our graph logic
                "thoughts": thoughts,
                "output_json": res_dict,
            }
            
            # 4. Append to dataset
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            logger.info(f"Distilled 1 reasoning pair -> {self.log_path.name}")
            
        except Exception as e:
            logger.error(f"Distillation Capture Failed: {e}")

    def finalize(self):
        """Final cleanup if needed."""
        logger.info("Distillation Session Finalized.")
