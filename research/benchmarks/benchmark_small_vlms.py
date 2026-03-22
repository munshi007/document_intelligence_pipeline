import time
import logging
from PIL import Image
from common.vlm_client import VLMClient
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class DocumentInsight(BaseModel):
    title: str = Field(description="Main title of the document")
    technical_specs: List[str] = Field(description="List of 3-5 technical specifications found")
    summary: str = Field(description="One sentence summary of the page context")

    @field_validator('technical_specs', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        # If a "lazy" small model returns a string instead of a list, fix it!
        if isinstance(v, str):
            return [v]
        # If it returns a dict (common in some 8B models), flatten it!
        if isinstance(v, dict):
            return [str(val) for val in v.values()]
        return v

def benchmark_models():
    # Candidates for the "Small but Mighty" Leaderboard
    MODELS = [
        {"provider": "ollama", "model": "qwen3-vl:8b", "label": "Qwen3-VL (8B)"}
    ]
    
    sample_path = "research/dataset/images/54620_hdb_de_15_p001.jpg"
    image = Image.open(sample_path)
    
    print("\n" + "="*80)
    print(f"{'VLM LEADERBOARD COMPARISON':^80}")
    print("="*80)
    print(f"{'Model':<30} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 80)

    for entry in MODELS:
        # 1. Unload all first to be extra safe
        if entry["provider"] == "ollama":
             import requests
             requests.post("http://localhost:11434/api/generate", json={"model": entry["model"], "keep_alive": 0})
             time.sleep(2)

        vlm = VLMClient({
            "provider": entry["provider"],
            "model": entry["model"]
        })
        
        start_time = time.time()
        try:
            result = vlm.generate_structured(
                image=image,
                prompt="Extract technical insights from this industrial document page.",
                response_model=DocumentInsight
            )
            elapsed = time.time() - start_time
            
            if result:
                status = "✅ SUCCESS"
                print(f"{entry['label']:<30} | {status:<10} | {elapsed:<10.2f}")
            else:
                status = "❌ FAIL"
                print(f"{entry['label']:<30} | {status:<10} | {elapsed:<10.2f}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{entry['label']:<30} | ❌ ERROR | {elapsed:<10.2f}")
        finally:
            # 2. Force unload after each run
            if entry["provider"] == "ollama":
                requests.post("http://localhost:11434/api/generate", json={"model": entry["model"], "keep_alive": 0})
                time.sleep(3) # Wait for VRAM to settle

    print("="*80)

if __name__ == "__main__":
    benchmark_models()
