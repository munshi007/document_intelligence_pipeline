"""
Enhanced PDF Processing Pipeline - Main Entry Point
Modular architecture with advanced layout detection and table processing
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF

from config import setup_environment, get_output_paths, LOGGING_CONFIG
from pipeline import EnhancedPipeline

# Setup environment
setup_environment()

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger('enhanced_pdf_pipeline')

def main():
    """Main function to run the enhanced PDF pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced PDF Processing Pipeline - Modular Architecture')
    parser.add_argument('pdf_path', nargs='?', default="PDF/Super_Complex_2.pdf", 
                       help='Path to PDF file')
    parser.add_argument('--output', '-o', default=None, 
                       help='Output directory (if not specified, creates numbered subdirectory in Output/)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with detailed visualizations')
    parser.add_argument('--debug-page', type=int, 
                       help='Process only specific page number for debugging')
    
    parser.add_argument('--max-pages', type=int, default=None,
                       help='Maximum number of pages to process')
    parser.add_argument('--pages', type=str, default=None,
                       help='Specific pages to process (e.g., "1-5", "1,3,5")')
    
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    output_dir = args.output
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        logger.info("Available PDFs:")
        pdf_dir = Path("PDF")
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                logger.info(f"  - {pdf_file}")
        return
    
    try:
        # Setup output directories
        output_paths = get_output_paths(output_dir)
        for key, path in output_paths.items():
            if isinstance(path, Path) and key != 'json_file' and key != 'markdown_file':
                path.mkdir(parents=True, exist_ok=True)
        
        if args.debug:
            logger.info("Debug mode enabled - generating detailed visualizations")
        
        if args.debug_page:
            logger.info(f"Debug mode: processing only page {args.debug_page}")
        
        # Initialize pipeline once (model reuse)
        pipeline = EnhancedPipeline(output_dir=output_dir, debug_mode=args.debug)
        
        # Process PDF
        result = pipeline.process_pdf(
            pdf_path, 
            max_pages=args.max_pages,
            page_range=args.pages
        )
        
        # Print summary
        if result and "summary" in result:
            print_summary(result, pipeline)
        else:
            logger.error("Processing returned no results.")
        
    except Exception as e:
        logger.error(f"Enhanced pipeline execution failed: {e}")
        raise

# Helper functions moved to EnhancedPipeline class

def print_summary(result: dict, pipeline):
    """Print processing summary to console."""
    print("\n" + "="*80)
    print("ENHANCED PDF LAYOUT PROCESSING COMPLETE")
    print("="*80)
    print(f"Document: {result['document_info']['filename']}")
    print(f"Pages processed: {result['summary']['pages_processed']}")
    print(f"Total regions detected: {result['summary']['total_regions']}")
    print(f"Tables found: {result['summary']['tables_found']}")
    print()
    print("Region Types:")
    for region_type, count in result['summary']['region_types'].items():
        print(f"  - {region_type}: {count}")
    print()
    print("Processing Methods:")
    for method, count in result['summary']['processing_methods'].items():
        print(f"  - {method}: {count}")
    print()
    print("Output Files:")
    print(f"  - Layout data: {pipeline.output_dir / 'enhanced_layout_blocks.json'}")
    print(f"  - Markdown content: {pipeline.output_dir / 'extracted_content.md'}")
    print(f"  - Thumbnails: {pipeline.thumbnails_dir}/")
    print("="*80)
    
    if result['summary']['pages_with_errors'] > 0:
        print(f"⚠️  Pages with errors: {result['summary']['pages_with_errors']}")

if __name__ == "__main__":
    main()