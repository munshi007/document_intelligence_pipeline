"""
Extraction-Only Runner
======================
Used to bypass GPU/OCR hangs by running the Librarian Agent 
directly on a previously stored DocumentGraph.
"""

import sys
import json
import logging
from pathlib import Path

# Add core directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from extractor.agent import ExtractorAgent
from extractor.schema_engine import SchemaAuditor
from extractor.schema_definitions import LibrarianUniversalHardware
from storage.store import EphemeralStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extra_only")

def main():
    graph_path = Path("output/v3/storage/Super_Complex_2_graph.json")
    if not graph_path.exists():
        print(f"Error: Graph not found at {graph_path}")
        return

    # 1. Load Graph
    logger.info(f"Loading stored HKG graph: {graph_path}")
    store = EphemeralStore(storage_dir="output/v3/storage")
    # Cheat: just load the JSON directly using Pydantic
    from core.schemas import DocumentGraph
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = DocumentGraph.model_validate_json(f.read())

    logger.info(f"Loaded {len(graph.nodes)} nodes.")

    # 2. Audit Phase
    auditor = SchemaAuditor(model_id="gpt-4o")
    audit_results = auditor.audit_document(graph)
    discovery = auditor.get_discovery_schema(audit_results)
    logger.info(f"Discovered Modules: {discovery['active_modules']}")

    # 3. Librarian Agentic Extraction
    logger.info("Executing Librarian Agentic Extraction...")
    extractor = ExtractorAgent(model_id="gpt-4o")
    
    # We take the first 10 nodes for this test run
    context_markdown = "\n\n".join([n.content for n in graph.nodes[:10]])
    result = extractor.extract_structured(
        image=None,
        prompt="Perform industrial hardware extraction.",
        response_model=LibrarianUniversalHardware,
        domain="Hardware",
        context_markdown=context_markdown,
    )
    
    # 4. Save Final Output
    out_path = Path("output/v3/Super_Complex_2_universal_extraction_SOTA.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)
        
    logger.info(f"SUCCESS: Extraction saved to {out_path}")
    print("\n--- FINAL MASTER RECORD ---")
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()
