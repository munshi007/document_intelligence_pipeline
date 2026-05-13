import sys
import os
import logging

# ==============================================================================
# SMART LOGGING: Silence library-level noise before anything else is initialized
# ==============================================================================
logging.getLogger("unsloth").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("trl").setLevel(logging.CRITICAL)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
# ==============================================================================

import argparse
from pathlib import Path
import json
import shutil

# Add core directory to sys.path
sys.path.append(str(Path(__file__).parent))

from converter.engine import ConverterEngine
from chunker.graph_builder import GraphBuilder
from storage.store import EphemeralStore
from extractor.agent import ExtractorAgent, ExtractionFailureError
from extractor.schema_engine import SchemaAuditor
from extractor.discovery_agent import DiscoveryAgent, DiscoveryResult
from extractor.schema_registry import get_schema_model, list_schema_families
from extractor.evaluation import evaluate_extraction
from extractor.normalizer import FieldNormalizer


def _load_explicit_schema(schema_path: str) -> dict:
    """Load an explicit runtime schema JSON file."""
    if not schema_path:
        raise ValueError("--schema_path is required when --schema_mode explicit")
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Explicit schema not found: {schema_file}")
    with open(schema_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Silence common noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Orchestrator v3: Librarian-Grade Document Intelligence")
    parser.add_argument("pdf_path", type=str, help="Path to the source PDF file")
    parser.add_argument("--output_dir", type=str, default="output/v3", help="Base output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_pages", type=int, help="Limit number of pages")

    # Extraction options
    parser.add_argument("--extract", action="store_true", help="Enable extraction step")
    parser.add_argument("--auto_schema", action="store_true", help="(Deprecated) Enables discovery route. Use --schema_mode auto|domain")
    parser.add_argument("--schema_mode", type=str, choices=["auto", "domain", "explicit"], default="auto", help="Schema routing mode")
    parser.add_argument("--schema_path", type=str, default=None, help="Path to explicit extraction schema JSON (used with --schema_mode explicit)")
    parser.add_argument("--save_debug_traces", action="store_true", help="Create debug trace directory for extraction diagnostics")
    parser.add_argument("--evaluate", action="store_true", help="Reserved flag for evaluation stage")
    parser.add_argument("--extractor_model", type=str, default="RMunshi/librarian-qwen-extractor", help="Text model for extraction")
    parser.add_argument("--model", type=str, default="RMunshi/vlm-student-thesis", help="Vision model for layout parsing (default: local Llama fine-tune)")
    parser.add_argument("--distill", action="store_true", help="Enable distillation (data capture)")
    parser.add_argument("--with_grounding", action="store_true", help="Enable langextract-based precision grounding")

    # Ablation control flags
    parser.add_argument("--no_routing", action="store_true",
                        help="Ablation A0: always use general_v1, skip heuristic routing")
    parser.add_argument("--no_projection", action="store_true",
                        help="Ablation A0/A1: skip project_to_schema() step")
    parser.add_argument("--with_normalization", action="store_true",
                        help="Ablation A3: apply FieldNormalizer to projected payload")

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("orchestrator_v3")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_traces:
        (output_dir / "debug_traces").mkdir(parents=True, exist_ok=True)

    doc_stem = Path(args.pdf_path).stem

    # ── STEP 1: Vision-to-Primal ─────────────────────────────────────
    logger.info(f">>> STEP 1: Converting PDF → Primal Regions (VLM: {args.model})")
    converter = ConverterEngine(output_dir=args.output_dir, debug=args.debug, vlm_model=args.model)
    regions = converter.convert_to_regions(args.pdf_path, max_pages=args.max_pages)

    # ── STEP 2: Structured Markdown + Manifest ───────────────────────
    logger.info(">>> STEP 2: Building Structured Markdown + Manifest")
    doc_info = {
        "doc_id": doc_stem,
        "filename": Path(args.pdf_path).name,
        "total_pages": len(set(r.page for r in regions)),
    }
    markdown, manifest = converter.build_markdown_and_manifest(regions, doc_info)

    # Save outputs
    md_path = output_dir / "extracted_content.md"
    with open(md_path, 'w', encoding='utf-8') as f: f.write(markdown)
    
    manifest_path = output_dir / f"{doc_stem}_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)

    # ── STEP 3: Hierarchical Knowledge Graph (HKG) ───────────────────
    logger.info(">>> STEP 3: Building Hierarchical Knowledge Graph (HKG)")
    chunker = GraphBuilder()
    graph = chunker.build_graph(regions, doc_info)
    
    storage = EphemeralStore(storage_dir=str(output_dir / "storage"))
    storage.save_graph(graph)

    # Human-readable graph summary
    graph_summary_path = output_dir / f"{doc_stem}_graph_summary.txt"
    with open(graph_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Document Graph Summary: {doc_info['filename']}\n")
        f.write("=" * 80 + "\n")
        for node in graph.nodes:
            f.write(f"[{node.node_id}] {node.metadata.get('breadcrumb', 'Root')}\n")
            f.write(f"   Content: {node.content[:200]}...\n\n")

    # ── STEP 4: Adaptive Extraction ──────────────────────────────────
    if args.extract:
        # GPU FLUSH: Free up VRAM from the Vision model (Phase 1) before loading the 15GB Text Extractor
        import torch
        import gc
        logger.info(">>> GPU MEMORY: Clearing Vision model from VRAM...")
        
        # SOTA: Correct path to the VLM provider in the modular architecture
        vlm_client = getattr(converter.pipeline, 'components', {}).get('vlm_client')
        if vlm_client and hasattr(vlm_client, 'provider'):
            provider = vlm_client.provider
            # If it's a local provider, clear its class-level or instance-level model references
            if hasattr(provider, '_model'):
                type(provider)._model = None
                type(provider)._tokenizer = None
            elif hasattr(provider, 'model'):
                provider.model = None
                provider.tokenizer = None
        
        # Clear any other large layout models (YOLO, TATR)
        if hasattr(converter, 'pipeline'):
            converter.pipeline.components.clear()
            
        del converter
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f">>> STEP 4: Agentic Extraction ({args.extractor_model})")


        # 1. Discovery Phase (Scout + Architect)
        discovery_agent = DiscoveryAgent(model_id=args.extractor_model)
        
        if args.schema_mode == "explicit" and args.schema_path:
            logger.info(f"    Schema Mode: EXPLICIT (Path: {args.schema_path})")
            response_model = discovery_agent.synthesize_from_external_schema(args.schema_path)
            domain = response_model.__name__
            is_high_density = True # Standard for technical schemas
            
            # Create a dummy discovery result for consistency
            discovery_result = DiscoveryResult(
                domain=domain,
                is_high_density=is_high_density,
                dynamic_fields=[],
                confidence=1.0
            )
        else:
            doc_preview = markdown[:10000]
            # Read the graph summary to provide structural context
            graph_summary = ""
            if graph_summary_path.exists():
                with open(graph_summary_path, 'r', encoding='utf-8') as f:
                    graph_summary = f.read()
                    
            discovery_result = discovery_agent.scout(doc_preview, graph_summary=graph_summary)
            # Architect: Synthesize the custom Pydantic model
            response_model = discovery_agent.synthesize_model(discovery_result)
            domain = discovery_result.domain
            is_high_density = discovery_result.is_high_density
            
            # Save the synthesized schema for user audit
            auto_schema_path = output_dir / f"{doc_stem}_auto_schema.json"
            with open(auto_schema_path, 'w', encoding='utf-8') as f:
                json.dump(response_model.model_json_schema(), f, indent=2, ensure_ascii=False)
            logger.info(f"    Saved Auto-Generated Schema: {auto_schema_path}")

        logger.info(f"    Discovery Domain: {domain}")
        if hasattr(discovery_result, 'nested_skeleton') and discovery_result.nested_skeleton:
            logger.info(f"    Nested Skeleton synthesized successfully.")

        # Save discovery for audit
        discovery_path = output_dir / f"{doc_stem}_discovery.json"
        with open(discovery_path, 'w', encoding='utf-8') as f:
            json.dump(discovery_result.model_dump(), f, indent=2, ensure_ascii=False)

        # 2. Distillation Setup
        observer = None
        if args.distill:
            from research.distillation_agent import DistillationAgent
            observer = DistillationAgent(dataset_dir="research/dataset_extraction")

        # 3. Execution Phase
        agent = ExtractorAgent(model_id=args.extractor_model, observer=observer)
        ext_path = output_dir / f"{doc_stem}_universal_extraction.json"
        
        try:
            logger.info(f"    Starting Extraction (Grounding={args.with_grounding})...")
            
            # Pre-fill sourceFile if it's in the schema
            initial_data = {}
            if "sourceFile" in response_model.model_fields:
                initial_data["sourceFile"] = Path(args.pdf_path).name
            
            final_record = agent.extract_structured(
                image=None, 
                prompt="Extract all factual data according to the provided schema.",
                response_model=response_model,
                domain=domain,
                is_high_density=is_high_density,
                context_markdown=markdown,
                context_nodes=graph.nodes,
                use_grounding=args.with_grounding,
                trace_context={"trace_dir": str(output_dir / "debug_traces")} if args.save_debug_traces else None
            )
            
            # Merge pre-filled data if model didn't overwrite with something valid
            if initial_data:
                for k, v in initial_data.items():
                    if not getattr(final_record, k, None):
                        setattr(final_record, k, v)

            # ── STEP 5: Finalization ─────────────────────────────────────────
            logger.info(">>> STEP 5: Finalizing Output")

            with open(ext_path, 'w', encoding='utf-8') as f:
                json.dump(final_record.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info(f"    Extraction Complete: {ext_path}")

            # Write grounding audit (if the extractor produced stats)
            gr_stats = getattr(agent, "_last_grounding_stats", None)
            if gr_stats is not None:
                grounding_path = output_dir / f"{doc_stem}_grounding.json"
                with open(grounding_path, 'w', encoding='utf-8') as f:
                    json.dump(gr_stats, f, indent=2, ensure_ascii=False)
                rate = gr_stats["verified"] / max(gr_stats["checked"], 1) if gr_stats["checked"] else 0.0
                retries_attempted = gr_stats.get("retries_attempted", 0)
                retries_accepted = gr_stats.get("retries_accepted", 0)
                retry_bit = (
                    f", retries={retries_accepted}/{retries_attempted}"
                    if retries_attempted else ""
                )
                logger.info(
                    f"    Verbatim grounding: {gr_stats['verified']}/{gr_stats['checked']} "
                    f"= {rate:.2f} (repaired={len(gr_stats['repaired'])}, "
                    f"flagged={len(gr_stats['flagged'])}{retry_bit})"
                )

        except ExtractionFailureError as e:
            logger.error(f"Extraction failed: {e}")
            with open(ext_path, 'w', encoding='utf-8') as f:
                json.dump({"status": "failed", "error": str(e)}, f, indent=2)

        # ── STEP 5b: Evaluation Scorecard (always on when extraction ran) ──
        logger.info(">>> STEP 5b: Computing Evaluation Scorecard")
        try:
            gt = None
            gt_candidates = [
                Path(args.pdf_path).parent / f"{doc_stem}.gt.json",
                Path("data/ground_truth") / f"{doc_stem}.gt.json",
            ]
            for gt_path in gt_candidates:
                if gt_path.exists():
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        gt = json.load(f)
                    logger.info(f"    Ground truth loaded: {gt_path}")
                    break

            with open(ext_path, 'r', encoding='utf-8') as f:
                extraction_payload = json.load(f)

            scorecard = evaluate_extraction(
                extraction_payload=extraction_payload,
                target_schema_json=response_model.model_json_schema(),
                doc_stem=doc_stem,
                trace_dir=str(output_dir / "debug_traces") if args.save_debug_traces else None,
                ground_truth=gt,
                grounding_stats=getattr(agent, "_last_grounding_stats", None),
            )
            eval_path = output_dir / f"{doc_stem}_eval.json"
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(scorecard, f, indent=2, ensure_ascii=False)

            gt_bit = f", F1={scorecard.get('gt_field_f1')}" if gt else ""
            logger.info(
                f"    Scorecard: status={scorecard['status']}, "
                f"populated_rate={scorecard.get('non_empty_extraction_rate', 0):.2f}, "
                f"required_completion={scorecard.get('required_field_completion_rate', 0):.2f}"
                f"{gt_bit}"
            )
            logger.info(f"    Saved Scorecard: {eval_path}")
        except Exception as eval_err:
            logger.warning(f"Evaluation skipped due to error: {eval_err}")

    else:
        logger.info(">>> STEP 4: Extraction SKIPPED.")

    print("\n" + "=" * 80)
    print("V3 LIBRARIAN PIPELINE COMPLETE")
    print(f"  Graph Summary:  {graph_summary_path}")
    if args.extract:
        print(f"  Final Record:   {ext_path}")
        eval_path_print = output_dir / f"{doc_stem}_eval.json"
        if eval_path_print.exists():
            print(f"  Scorecard:      {eval_path_print}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
