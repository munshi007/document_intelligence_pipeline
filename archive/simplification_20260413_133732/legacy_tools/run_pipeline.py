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
    
    parser.add_argument("--complete", help="Run end-to-end (PDF -> MD -> Chunks -> Schema Extraction)", action="store_true")
    parser.add_argument("--schema", help="Path to JSON schema for extraction", default="schema_sample.json")
    
    args = parser.parse_args()
    
    input_path = Path(args.pdf_path)
    
    # Modular approach if --complete is set
    if args.complete:
        run_complete_pipeline(input_path, args)
        return

    # Standard / Legacy processing...
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

def run_complete_pipeline(pdf_path: Path, args):
    """Orchestrate the modular tools sequentially."""
    import subprocess
    
    pdf_name = pdf_path.stem[:50]
    output_dir = f"output/toolkit_{pdf_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. PDF to Markdown
    logger.info(">>> Step 1: Running PDF-to-Markdown (tool_pdf2md.py)")
    cmd1 = [sys.executable, "tool_pdf2md.py", str(pdf_path), "--output_dir", output_dir]
    if args.debug: cmd1.append("--debug")
    if args.max_pages: cmd1.extend(["--max_pages", str(args.max_pages)])
    subprocess.run(cmd1, check=True)
    
    # 2. Chunking
    layout_json = Path(output_dir) / "enhanced_layout_blocks.json"
    logger.info(">>> Step 2: Running Layout-Aware Chunking (tool_chunker.py)")
    cmd2 = [sys.executable, "tool_chunker.py", str(layout_json)]
    subprocess.run(cmd2, check=True)
    
    # 3. Extraction
    chunks_json = Path(output_dir) / "extracted_chunks.json"
    logger.info(">>> Step 3: Running Schema Extraction (tool_extractor.py)")
    cmd3 = [sys.executable, "tool_extractor.py", str(chunks_json), "--schema_path", args.schema]
    subprocess.run(cmd3, check=True)
    
    print("\n" + "="*80)
    print("TOOLKIT COMPLETE: End-to-End processing finished!")
    print(f"Final Data: {Path(output_dir) / 'final_structured_data.json'}")
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    main()
