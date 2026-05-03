#!/usr/bin/env python3
import json
import argparse
import logging
import sys
from pathlib import Path

# Add core directory to sys.path
sys.path.append(str(Path(__file__).parent))

from processors.layout_chunker import LayoutChunker

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Tool 2: Layout-Aware Hierarchical Chunker")
    parser.add_argument("layout_json_path", type=str, help="Path to the layout JSON file produced by Tool 1")
    parser.add_argument("--output_path", type=str, help="Path to save the chunks JSON file")
    parser.add_argument("--max_chars", type=int, default=2500, help="Maximum characters per chunk")
    
    args = parser.parse_args()
    setup_logging()
    
    logger = logging.getLogger("chunker")
    logger.info(f"Loading Layout JSON: {args.layout_json_path}")
    
    with open(args.layout_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pages = data.get('pages', [])
    chunker = LayoutChunker(max_chars=args.max_chars)
    
    all_chunks = []
    for page in pages:
        page_num = page.get('page_num')
        regions = page.get('regions', [])
        logger.info(f"Chunking Page {page_num} with {len(regions)} regions...")
        
        page_chunks = chunker.chunk_regions(regions, page_num)
        all_chunks.extend(page_chunks)
        
    # Determine output path
    output_path = args.output_path
    if not output_path:
        base_dir = Path(args.layout_json_path).parent
        output_path = base_dir / "extracted_chunks.json"
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("Layout-Aware Chunking Completed Successfully!")
    print(f"Chunks Saved: {output_path}")
    print(f"Total Chunks Generated: {len(all_chunks)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
