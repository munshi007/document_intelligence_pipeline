"""
Master Distiller: Librarian High-Throughput Processing
======================================================
Automated loop to process the CURATED_DATA library (945 PDFs).
Captures SFT-ready training data for local student models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import traceback

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from run_v3 import setup_logging
from converter.engine import ConverterEngine
from chunker.graph_builder import GraphBuilder
from extractor.agent import ExtractorAgent
from extractor.schema_engine import SchemaAuditor
from research.distillation_agent import DistillationAgent
from extractor.schema_registry import get_schema_for_domain

def main():
    parser = argparse.ArgumentParser(description="Librarian Batch Distiller")
    parser.add_argument("--source", type=str, default="data/CURATED_DATA", help="Path to source PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N PDFs")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Teacher model")
    parser.add_argument("--pages", type=int, default=3, help="Max pages per PDF to sample")
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("LibrarianDistiller")
    
    source_dir = Path(args.source)
    pdfs = sorted(list(source_dir.glob("*.pdf")))
    
    logger.info(f">>> Librarian Distiller: Starting batch on {len(pdfs)} PDFs.")
    if args.limit:
        pdfs = pdfs[:args.limit]
        logger.info(f"    Restricted to first {args.limit} samples.")

    # 1. Initialize Components
    distiller = DistillationAgent(dataset_dir="research/distilled_labels")
    agent = ExtractorAgent(model_id=args.model, observer=distiller)
    scout = SchemaAuditor(model_id=args.model)
    converter = ConverterEngine(output_dir="output/distillation_temp", debug=False)
    graph_builder = GraphBuilder()

    completed = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs):
        doc_id = pdf_path.stem
        logger.info(f"\n[{i+1}/{len(pdfs)}] DISTILLING: {doc_id}")
        
        try:
            # Stage 1: Conversion (Local Vision)
            regions = converter.convert_to_regions(str(pdf_path), max_pages=args.pages)
            doc_info = {"doc_id": doc_id, "filename": pdf_path.name, "total_pages": len(regions)}
            
            # Stage 2: Graph Construction
            graph = graph_builder.build_graph(regions, doc_info)
            
            # Stage 3: Scout Audit (Teacher Domain Discovery)
            audit = scout.audit_document(graph)
            discovery = scout.get_discovery_schema(audit)
            domain = discovery['domain']
            schema_class_name = discovery['base_model']
            
            # Dynamically get the Pydantic model
            from extractor.schema_definitions import LibrarianUniversalHardware, LibrarianGeneralClerk
            if schema_class_name == "LibrarianUniversalHardware":
                schema_class = LibrarianUniversalHardware
            else:
                schema_class = LibrarianGeneralClerk

            logger.info(f"    Librarian Scout: Domain identified as [{domain}] -> using {schema_class_name}")

            # Stage 4: Execution (Teacher Extraction + Thought Capture)
            result = agent.extract_structured(
                graph.nodes, 
                response_model=schema_class, 
                domain=domain
            )
            
            logger.info(f"    SUCCESS: Captured {len(graph.nodes)} nodes into training set.")
            completed += 1
            
        except Exception as e:
            logger.error(f"    FAILED: {doc_id} -> {e}")
            traceback.print_exc()
            failed += 1
            continue

    logger.info(f"\n" + "=" * 60)
    print(f"DISTILLATION COMPLETE")
    print(f"  Total Processed: {len(pdfs)}")
    print(f"  Success:         {completed}")
    print(f"  Failures:        {failed}")
    print(f"  Dataset:         {distiller.log_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
