import logging
import json
import requests
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class VLMClient:
    """Client for interacting with local Ollama VLM (Qwen2-VL)."""
    
    def __init__(self, model_name: str = "qwen3-vl:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_chat = f"{base_url}/api/chat"
        
    def check_availability(self) -> bool:
        # (Same as before)
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                is_available = any(self.model_name in m for m in models)
                if is_available:
                    logger.info(f"VLM Model '{self.model_name}' is available.")
                    return True
                else:
                    logger.warning(f"VLM Model '{self.model_name}' not found. Available: {models}")
                    return False
            return False
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}")
            return False

    def verify_layout(self, image_data: bytes, regions: List[Dict]) -> List[int]:
        """
        Ask VLM to verify layout regions using Chat API.
        """
        # Convert image to base64
        img_b64 = base64.b64encode(image_data).decode('utf-8')
        
        region_desc = []
        for r in regions:
            bbox_str = f"[{int(r['bbox'][0])}, {int(r['bbox'][1])}, {int(r['bbox'][2])}, {int(r['bbox'][3])}]"
            region_desc.append(f"ID {r['region_id']} ({r['type']}): {bbox_str}. Text: '{r.get('text', '')[:20]}...'")
            
        system_msg = (
            "You are a document layout validator. Output ONLY valid JSON."
        )
        
        user_msg = (
            "Analyze the document image and these detected regions:\n"
            + "\n".join(region_desc) + "\n\n"
            "Identify the valid regions that capture real content (Tables, Diagrams, Paragraphs).\n"
            "Ignore noise or duplicate small text boxes inside tables.\n"
            "Return JSON with 'keep_ids': list[int].\n"
            "Example: {\"keep_ids\": [0, 2]}"
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": user_msg,
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "format": "json", # Enforce JSON Output
            "options": {
                "temperature": 0.0 # Deterministic
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.api_chat, json=payload, timeout=300)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                # Chat API returns 'message': {'content': ...}
                result = response.json().get('message', {}).get('content', '')
                logger.info(f"VLM Response ({duration:.2f}s): {result}")
                
                # Cleanup and Parse
                cleaned_result = result.strip()
                if cleaned_result.startswith("```"):
                     lines = cleaned_result.split('\n')
                     if len(lines) >= 3:
                         cleaned_result = "\n".join(lines[1:-1])
                
                try:
                    data = json.loads(cleaned_result)
                    keep_ids = data.get('keep_ids', [])
                    # Handle "ensemble_X" or "layoutparser_Y" IDs ?
                    # The prompt asked for IDs. If regions passed in have numeric indices, we can use those.
                    # Current regions have 'region_id' which is string "ensemble_X".
                    # I should probably map them to simple integers for the LLM prompt to avoid hallucination.
                    return keep_ids
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse VLM JSON response: {e}")
                    print(f"DEBUG: RAW VLM RESPONSE:\n{result}\n-------------------")  # Direct debug output
                    return []
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"VLM Inference failed: {e}")
            return []

    def verify_layout_smart(self, image_data: bytes, regions: List[Dict]) -> List[str]:
        """
        Wrapper that handles ID mapping for the LLM.
        Returns list of original region_ids strings to keep.
        """
        # Map complex string IDs to simple integers 0..N
        # This makes it easier for the LLM to output valid JSON
        id_map = {i: r['region_id'] for i, r in enumerate(regions)}
        
        # Prepare regions for prompt with simple integer IDs
        simple_regions = []
        for i, r in enumerate(regions):
            simple_regions.append({
                'region_id': i, # Simple Integer
                'type': r['type'],
                'bbox': r['bbox'],
                'text': r.get('text', '')
            })
            
        # Call VLM
        keep_indices = self.verify_layout(image_data, simple_regions)
        
        # Map back to string IDs
        keep_string_ids = []
        for idx in keep_indices:
            if idx in id_map:
                keep_string_ids.append(id_map[idx])
                
        # If VLM fails/returns empty, we might want to default to keeping everything 
        # (fail open) rather than deleting everything.
        if not keep_string_ids:
             logger.warning("VLM returned no valid IDs. Falling back to keeping all.")
             return list(id_map.values())
             
        # Calculate what was dropped for logging
        dropped = set(id_map.values()) - set(keep_string_ids)
        if dropped:
            logger.info(f"VLM Agent dropped {len(dropped)} regions: {dropped}")
            
        return keep_string_ids
