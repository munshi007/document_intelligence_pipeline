"""
Stage 5 — extract.

Reads:   <doc_stem>_auto_schema.json (Pydantic schema → reconstructed via
         DiscoveryAgent.synthesize_from_external_schema)
         <doc_stem>_discovery.json   (domain, is_high_density)
         extracted_content.md        (full markdown context)
         storage/<doc_stem>_graph.json (graph nodes via EphemeralStore.load_graph)
Writes:  <doc_stem>_universal_extraction.json
         <doc_stem>_grounding.json    (if grounding produced stats)
         <doc_stem>_eval.json         (always, when extraction ran)

Cost: loads the 15 GB text extractor. Reuses an already-loaded model if
discovery and extract run in the same Python process.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from extractor.agent import ExtractorAgent, ExtractionFailureError
from extractor.discovery_agent import DiscoveryAgent
from extractor.evaluation import evaluate_extraction
from storage.store import EphemeralStore
from stages.paths import StagePaths

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTOR = "RMunshi/librarian-qwen-extractor"


def _load_discovery_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Discovery artifact missing: {path}. Run discover-schema first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ground_truth(pdf: Path, doc_stem: str) -> Optional[Dict[str, Any]]:
    for gt_path in (
        pdf.parent / f"{doc_stem}.gt.json",
        Path("data/ground_truth") / f"{doc_stem}.gt.json",
    ):
        if gt_path.exists():
            logger.info(f"[extract] ground truth loaded: {gt_path}")
            return json.loads(gt_path.read_text(encoding="utf-8"))
    return None


def run_extract(
    pdf: Path,
    paths: StagePaths,
    *,
    extractor_model: str = DEFAULT_EXTRACTOR,
    with_grounding: bool = False,
    save_debug_traces: bool = False,
    distill: bool = False,
    force: bool = False,
) -> None:
    paths.ensure()

    if paths.extraction.exists() and not force:
        logger.info(f"[extract] reusing existing {paths.extraction.name}")
        return

    # Reconstruct the runtime Pydantic schema from the auto_schema JSON. This is
    # the discovery→extract handoff: a single function that already exists on
    # DiscoveryAgent and produces an identical model to synthesize_model().
    if not paths.auto_schema.exists():
        raise FileNotFoundError(
            f"Schema artifact missing: {paths.auto_schema}. Run discover-schema first."
        )
    response_model = DiscoveryAgent(model_id=extractor_model).synthesize_from_external_schema(
        str(paths.auto_schema)
    )

    discovery_meta = _load_discovery_meta(paths.discovery)
    domain = discovery_meta.get("domain", "General")
    is_high_density = bool(discovery_meta.get("is_high_density", False))

    if not paths.markdown.exists():
        raise FileNotFoundError(
            f"Markdown artifact missing: {paths.markdown}. Run pdf-to-markdown first."
        )
    markdown = paths.markdown.read_text(encoding="utf-8")

    store = EphemeralStore(storage_dir=str(paths.graph_storage_dir))
    graph = store.load_graph(paths.doc_stem)
    if graph is None:
        raise FileNotFoundError(
            f"Graph artifact missing: {paths.graph_json}. Run md-to-graph first."
        )

    if save_debug_traces:
        paths.debug_traces.mkdir(parents=True, exist_ok=True)

    observer = None
    if distill:
        from research.distillation_agent import DistillationAgent
        observer = DistillationAgent(dataset_dir="research/dataset_extraction")

    agent = ExtractorAgent(model_id=extractor_model, observer=observer)

    initial_data: Dict[str, Any] = {}
    if "sourceFile" in response_model.model_fields:
        initial_data["sourceFile"] = pdf.name

    try:
        logger.info(
            f"[extract] domain={domain}, density={is_high_density}, grounding={with_grounding}"
        )
        final_record = agent.extract_structured(
            image=None,
            prompt="Extract all factual data according to the provided schema.",
            response_model=response_model,
            domain=domain,
            is_high_density=is_high_density,
            context_markdown=markdown,
            context_nodes=graph.nodes,
            use_grounding=with_grounding,
            trace_context=(
                {"trace_dir": str(paths.debug_traces)} if save_debug_traces else None
            ),
        )

        # Merge pre-filled defaults only when the model left the field empty.
        for k, v in initial_data.items():
            if not getattr(final_record, k, None):
                setattr(final_record, k, v)

        paths.extraction.write_text(
            json.dumps(final_record.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"[extract] wrote {paths.extraction.name}")

        gr_stats = getattr(agent, "_last_grounding_stats", None)
        if gr_stats is not None:
            paths.grounding.write_text(
                json.dumps(gr_stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            checked = gr_stats.get("checked", 0)
            verified = gr_stats.get("verified", 0)
            rate = verified / max(checked, 1) if checked else 0.0
            retries_a = gr_stats.get("retries_attempted", 0)
            retries_ok = gr_stats.get("retries_accepted", 0)
            retry_bit = f", retries={retries_ok}/{retries_a}" if retries_a else ""
            logger.info(
                f"[extract] grounding {verified}/{checked} = {rate:.2f} "
                f"(repaired={len(gr_stats.get('repaired', []))}, "
                f"flagged={len(gr_stats.get('flagged', []))}{retry_bit})"
            )

    except ExtractionFailureError as e:
        logger.error(f"[extract] extraction failed: {e}")
        paths.extraction.write_text(
            json.dumps({"status": "failed", "error": str(e)}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        # Pydantic ValidationError, downstream library errors, etc. The pipeline
        # contract is "always emit a well-formed extraction.json so callers don't
        # need to special-case missing files." Keep the error type in the payload
        # for easy triage; never let a process-killing exception escape this
        # boundary on the extraction path.
        logger.error(f"[extract] unexpected failure during extraction: {type(e).__name__}: {e}")
        paths.extraction.write_text(
            json.dumps(
                {"status": "failed", "error": f"{type(e).__name__}: {e}"[:2000]},
                indent=2,
            ),
            encoding="utf-8",
        )

    # Always emit an evaluation scorecard when extract ran.
    try:
        extraction_payload = json.loads(paths.extraction.read_text(encoding="utf-8"))
        scorecard = evaluate_extraction(
            extraction_payload=extraction_payload,
            target_schema_json=response_model.model_json_schema(),
            doc_stem=paths.doc_stem,
            trace_dir=str(paths.debug_traces) if save_debug_traces else None,
            ground_truth=_load_ground_truth(pdf, paths.doc_stem),
            grounding_stats=getattr(agent, "_last_grounding_stats", None),
        )
        paths.evaluation.write_text(
            json.dumps(scorecard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        gt_bit = (
            f", F1={scorecard.get('gt_field_f1')}"
            if scorecard.get("gt_field_f1") is not None
            else ""
        )
        logger.info(
            f"[extract] scorecard status={scorecard.get('status')}, "
            f"populated={scorecard.get('non_empty_extraction_rate', 0):.2f}, "
            f"required_completion={scorecard.get('required_field_completion_rate', 0):.2f}"
            f"{gt_bit} → {paths.evaluation.name}"
        )
    except Exception as eval_err:
        logger.warning(f"[extract] evaluation skipped: {eval_err}")
