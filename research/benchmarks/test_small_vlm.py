import logging
from PIL import Image
from common.vlm_client import VLMClient
from pydantic import BaseModel, Field
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleExtraction(BaseModel):
    title: str = Field(description="The main title of the document")
    keywords: List[str] = Field(description="Top 3 keywords from the document")

def test_small_vlm():
    # Using MiniCPM-V via Ollama - efficient and powerful (approx 4.5GB VRAM)
    SMALL_MODEL = "minicpm-v:latest"
    
    logger.info(f"🚀 Testing with SMALL VLM (Ollama): {SMALL_MODEL}")
    
    vlm = VLMClient({
        "provider": "ollama",
        "model": SMALL_MODEL
    })
    
    # Load a sample image
    try:
        sample_path = "research/dataset/images/54620_hdb_de_15_p001.jpg"
        image = Image.open(sample_path)
        
        logger.info("Extracting with 3B model...")
        result = vlm.generate_structured(
            image=image,
            prompt="Briefly analyze this document page.",
            response_model=SimpleExtraction
        )
        
        if result:
            print("\n" + "="*50)
            print("🎉 SMALL VLM EXTRACTION SUCCESSFUL!")
            print(f"Title: {result.title}")
            print(f"Keywords: {', '.join(result.keywords)}")
            print("="*50)
        else:
            logger.error("Extraction failed.")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_small_vlm()
