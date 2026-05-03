#!/usr/bin/env python3
import json
import argparse
import logging
import sys
import os
from pathlib import Path

# Add core directory to sys.path
sys.path.append(str(Path(__file__).parent))

try:
    import langextract as lx
except ImportError:
    print("Error: langextract not found. Please ensure it is installed in your environment.")
    sys.exit(1)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Tool 3: Schema-Conditioned LLM Extraction")
    parser.add_argument("chunks_json_path", type=str, help="Path to the chunks JSON file produced by Tool 2")
    parser.add_argument("--schema_path", type=str, help="Path to the target JSON schema file")
    parser.add_argument("--model_id", type=str, default="qwen2.5:7b", help="LLM Model ID (e.g., qwen2.5:7b)")
    parser.add_argument("--output_path", type=str, help="Path to save the final structured JSON")
    
    args = parser.parse_args()
    setup_logging()
    
    logger = logging.getLogger("extractor")
    logger.info(f"Loading Chunks: {args.chunks_json_path}")
    
    with open(args.chunks_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    # Load schema
    schema_path = args.schema_path or "schema_sample.json"
    if not os.path.exists(schema_path):
        logger.warning(f"Schema file {schema_path} NOT found. Creating a generic one.")
        sample_schema = {
            "title": "Document Information",
            "type": "object",
            "properties": {
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"}
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(sample_schema, f, indent=4)
            
    with open(schema_path, 'r', encoding='utf-8') as f:
        target_schema = json.load(f)
        
    logger.info(f"Using Model: {args.model_id}")
    logger.info(f"Target Schema: {target_schema.get('title', 'Generic Schema')}")
    
    # Process each chunk with LangExtract
    # We provide a default generic example as langextract requires it
    default_examples = [
        lx.data.ExampleData(
            text="This is a sample document about Project Alpha.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="project_name",
                    extraction_text="Project Alpha",
                    attributes={"confidence": "high"}
                )
            ]
        )
    ]
    
    all_extractions = []
    
    for i, chunk in enumerate(chunks):
        chunk_text = ""
        for r in chunk.get('regions', []):
            chunk_text += (r.get('text', '') or "") + "\n"
            
        logger.info(f"Extracting context for Chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars)...")
        
        try:
            # Using lx.extract with default examples
            result = lx.extract(
                text_or_documents=chunk_text,
                prompt_description=f"Extract key information based on the schema: {target_schema.get('title', 'Document Data')}",
                model_id=args.model_id,
                examples=default_examples # Required for LangExtract
            )
            
            # langextract typically returns a result object with .extractions
            all_extractions.append({
                "chunk_id": chunk.get('chunk_id'),
                "page_num": chunk.get('page_num'),
                "data": result.to_dict() if hasattr(result, 'to_dict') else str(result)
            })
        except Exception as e:
            logger.error(f"Error extracting from chunk {i}: {e}")
            all_extractions.append({
                "chunk_id": chunk.get('chunk_id'),
                "error": str(e)
            })
            
    # Determine output path
    output_path = args.output_path
    if not output_path:
        base_dir = Path(args.chunks_json_path).parent
        output_path = base_dir / "final_structured_data.json"
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extractions, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("Schema-Conditioned Extraction Completed Successfully!")
    print(f"Final Data Saved: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
