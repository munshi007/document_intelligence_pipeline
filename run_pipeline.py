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
    parser.add_argument("pdf_path", help="Path to the PDF file or directory to process")
    parser.add_argument("--pages", help="Page range to process (e.g., 1-3, 5)", default=None)
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    input_path = Path(args.pdf_path)
    
    # Batch directory processing
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
                debug_mode=args.debug
            )
            result = pipeline.process_pdf(
                str(pdf_path), 
                page_range=args.pages, 
                max_pages=args.max_pages
            )
            if result:
                print_summary(result, pipeline)
            logger.info(f"Successfully processed: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            if not input_path.is_dir(): # Re-raise if single file mode
                raise

def print_summary(result: dict, pipeline):
    """Print processing summary to console."""
    print("\n" + "="*80)
    print("PDF LAYOUT PROCESSING COMPLETE")
    print("="*80)
    print(f"Document: {result['document_info']['filename']}")
    print(f"Pages processed: {result['summary']['pages_processed']}")
    print(f"Total regions detected: {result['summary']['total_regions']}")
    print(f"Tables found: {result['summary']['tables_found']}")
    print("\nRegion Types:")
    for region_type, count in result['summary']['region_types'].items():
        print(f"  - {region_type}: {count}")
    print("\nProcessing Methods:")
    for method, count in result['summary']['processing_methods'].items():
        print(f"  - {method}: {count}")
    print("\nOutput Files:")
    print(f"  - Layout data: {pipeline.output_dir / 'enhanced_layout_blocks.json'}")
    print(f"  - Markdown content: {pipeline.output_dir / 'extracted_content.md'}")
    print(f"  - Thumbnails: {pipeline.thumbnails_dir}/")
    print("="*80)
    
    if result['summary']['pages_with_errors'] > 0:
        print(f"⚠️  Pages with errors: {result['summary']['pages_with_errors']}")

if __name__ == "__main__":
    main()
