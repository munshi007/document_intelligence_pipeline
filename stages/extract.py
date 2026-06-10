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
from typing import Any, Dict, List, Optional, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined as _PU

from extractor.agent import ExtractorAgent, ExtractionFailureError
from extractor.discovery_agent import DiscoveryAgent
from extractor.evaluation import evaluate_extraction
from storage.store import EphemeralStore
from stages.paths import StagePaths

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTOR = "RMunshi/librarian-qwen-extractor"


def _scrub_pydantic_undefined(obj: Any) -> Any:
    """Drop PydanticUndefined sentinels at any depth.

    ExtractorAgent's batch-merge logic occasionally builds partial sub-models
    via model_construct() (bypasses validation), leaving required leaves as
    PydanticUndefined. Pydantic v2's own json encoder refuses to serialize
    that sentinel, so model_dump() → json.dumps() blows up at the write site.
    Mirrors the scrub already used inside ExtractorAgent (agent.py near
    line 1565). Dict keys with a PU value are removed; PU inside lists is
    dropped from the list.
    """
    if obj is _PU:
        return None
    if isinstance(obj, dict):
        return {k: _scrub_pydantic_undefined(v) for k, v in obj.items() if v is not _PU}
    if isinstance(obj, list):
        return [_scrub_pydantic_undefined(v) for v in obj if v is not _PU]
    return obj


def _is_empty(v: Any) -> bool:
    """A dict-leaf is 'empty' if it carries no extracted information."""
    return v is None or v == "" or v == [] or v == {}


def _drop_invalid_list_items(data: Dict[str, Any], model_type: Type[BaseModel]) -> Dict[str, Any]:
    """Drop zombie sub-model entries from list-of-BaseModel fields.

    ExtractorAgent's batch merger occasionally emits a fully-empty sub-model
    via model_construct() — e.g. a TechParameter with every field None after
    the scrub. Re-validation against the inner type can't catch these because
    DiscoveryAgent.synthesize_model() builds dynamic sub-classes whose fields
    are all Optional (and so accept all-None). The reliable signal at the
    write boundary is therefore data-shaped, not schema-shaped: a dict whose
    every value is empty conveys no extraction and should be dropped.

    Only walks one level deep into list-of-BaseModel fields — that's the
    documented leak surface from agent.py's merger and matches what we see
    in practice.
    """
    cleaned = dict(data)
    for field_name, field_info in model_type.model_fields.items():
        if field_name not in cleaned:
            continue
        annotation = field_info.annotation
        if get_origin(annotation) not in (list, List):
            continue
        args = get_args(annotation)
        if not args or not isinstance(args[0], type) or not issubclass(args[0], BaseModel):
            continue
        inner_model = args[0]
        items = cleaned[field_name]
        if not isinstance(items, list):
            continue
        valid_items: List[Any] = []
        dropped = 0
        for item in items:
            if isinstance(item, dict) and all(_is_empty(v) for v in item.values()):
                dropped += 1
                continue
            valid_items.append(item)
        if dropped:
            logger.info(
                f"[extract] dropped {dropped} empty {inner_model.__name__} "
                f"item(s) from '{field_name}' (batch-merge zombie records)"
            )
        cleaned[field_name] = valid_items
    return cleaned


def _grounding_summary(gr_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a light, inline trust header from the verifier's audit.

    The full audit (provenance, per-field repairs, retry log) is persisted
    separately in <stem>_grounding.json. Here we surface only what a consumer
    of the extraction needs to judge trust at a glance: how many string spans
    were checked against the source, how many grounded, and which values could
    not be located there (``unverified`` — likely fabricated, since the
    extractor never sees the page image and so cannot have read them).

    Returns None when grounding did not run, so the caller can omit the key
    rather than emit a misleading all-zero block.
    """
    if not isinstance(gr_stats, dict) or not gr_stats.get("checked"):
        return None
    checked = int(gr_stats.get("checked", 0) or 0)
    verified = int(gr_stats.get("verified", 0) or 0)
    repaired_list = gr_stats.get("repaired", []) or []
    repaired = len(repaired_list)
    flagged_list = gr_stats.get("flagged", []) or []
    # case_restore repairs are already counted in `verified` (the value was
    # verbatim-grounded; only its casing was snapped to the source) — adding
    # them again would push pass_rate above 1.0.
    repaired_unverified = len(
        [r for r in repaired_list if r.get("kind") != "case_restore"]
    )
    grounded = verified + repaired_unverified
    return {
        "checked": checked,
        "verified": verified,
        "repaired": repaired,
        "flagged": len(flagged_list),
        "pass_rate": round(grounded / checked, 4) if checked else None,
        "unverified": [
            {
                "path": f.get("path"),
                "value": f.get("value"),
                "best_ratio": f.get("best_ratio"),
            }
            for f in flagged_list
        ],
    }


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
    response_model: Optional[Type[BaseModel]] = None,
) -> None:
    """Run the extraction stage.

    If `response_model` is provided (the in-process path, e.g. from
    stages.orchestrate.run_extract_group right after discovery), the live
    Pydantic class is used directly and no JSON-Schema round-trip happens.
    This matches what the historical run_v3.py did and preserves nested-type
    fidelity (e.g. SourceEvidence with typed page_number).

    If `response_model` is None (standalone subprocess invocation), we
    reconstruct the schema from auto_schema.json via
    synthesize_from_external_schema. That path is intentionally lossy — it
    drops $ref and anyOf details — but it remains the only sensible behavior
    when discovery ran in a different process and the only handoff available
    is the persisted JSON Schema.
    """
    paths.ensure()

    if paths.extraction.exists() and not force:
        logger.info(f"[extract] reusing existing {paths.extraction.name}")
        return

    if response_model is None:
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

        dumped = _scrub_pydantic_undefined(final_record.model_dump())
        if isinstance(dumped, dict):
            dumped = _drop_invalid_list_items(dumped, response_model)

        gr_stats = getattr(agent, "_last_grounding_stats", None)
        # Inline, non-destructive trust header: keep every extracted value, but
        # mark which ones the verifier could not locate in the source so a
        # downstream consumer never has to trust a value blind. Full audit still
        # goes to the <stem>_grounding.json sidecar below.
        if isinstance(dumped, dict):
            gsum = _grounding_summary(gr_stats)
            if gsum is not None:
                dumped["_grounding"] = gsum

        paths.extraction.write_text(
            json.dumps(dumped, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"[extract] wrote {paths.extraction.name}")

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
