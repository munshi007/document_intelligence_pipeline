import logging
import argparse
from pathlib import Path
from pipeline.enhanced_pipeline import EnhancedPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a")
    ],
    force=True
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Processing Pipeline")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--model", help="VLM model to use (e.g., gpt-4o, qwen2.5-vl:7b)", default=None)
    parser.add_argument("--provider", help="VLM provider (openai, ollama, internvl, got-ocr)", default=None)
    parser.add_argument("--strategy", choices=['gpt4o', 'sota_os', 'fast_os'], help="Research strategy (gpt4o, sota_os, fast_os)", default=None)
    parser.add_argument("--distill", help="Enable Self-Distillation (Capture Teacher data)", action="store_true")
    parser.add_argument("--pages", help="Page range to process (e.g., 1-3, 5)", default=None)
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    input_path = Path(args.pdf_path)
    
    # SOTA: Batch directory processing
    if input_path.is_dir():
        pdf_files = list(input_path.glob("**/*.pdf"))
        logger.info(f"Batch mode: Found {len(pdf_files)} PDFs in {input_path}")
    else:
        pdf_files = [input_path]

    for pdf_path in pdf_files:
        logger.info(f"--- Processing: {pdf_path.name} ---")
        pdf_name = pdf_path.stem[:50]
        output_dir = f"output/verification_{pdf_name}"
        
        try:
            pipeline = EnhancedPipeline(
                output_dir=output_dir, 
                debug_mode=args.debug,
                vlm_model=args.model,
                vlm_provider=args.provider,
                strategy=args.strategy,
                distill=args.distill
            )
            pipeline.process_pdf(
                str(pdf_path), 
                page_range=args.pages, 
                max_pages=args.max_pages
            )
            logger.info(f"Successfully processed: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            if not input_path.is_dir(): # Re-raise if single file mode
                raise

if __name__ == "__main__":
    main()
