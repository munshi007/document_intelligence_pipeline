#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to sys.path
sys.path.append(str(Path(__file__).parent))

from pipeline.enhanced_pipeline import EnhancedPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Tool 1: PDF to Layout-Aware Markdown")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--output_dir", type=str, default="output/latest", help="Output directory")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    setup_logging()
    
    logger = logging.getLogger("pdf2md")
    logger.info(f"Processing PDF: {args.pdf_path}")
    
    pipeline = EnhancedPipeline(output_dir=args.output_dir, debug_mode=args.debug)
    result = pipeline.process_pdf(args.pdf_path, max_pages=args.max_pages)
    
    markdown_path = pipeline.output_paths['markdown_file']
    json_path = pipeline.output_paths['json_file']
    
    print("\n" + "="*50)
    print("PDF-to-Markdown Completed Successfully!")
    print(f"Markdown: {markdown_path}")
    print(f"Layout JSON: {json_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
