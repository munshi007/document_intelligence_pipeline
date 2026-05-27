"""
Stage 4 — discover-schema.

Reads:   extracted_content.md, <doc_stem>_graph_summary.txt
         (or, in explicit mode, an external schema JSON path)
Writes:  <doc_stem>_discovery.json, <doc_stem>_auto_schema.json

Cost: loads the 15 GB text extractor on first construction of DiscoveryAgent.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from extractor.discovery_agent import DiscoveryAgent, DiscoveryResult
from stages.paths import StagePaths

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTOR = "RMunshi/librarian-qwen-extractor"

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _inject_hint_fields(
    schema_dict: Dict[str, Any],
    hint_fields: List[str],
) -> Dict[str, Any]:
    """Forcibly add each hint field as Optional[str] to schema_dict's properties.

    - Names already in `properties` are left untouched (discovered schema
      wins; preserves richer type info if discovery picked something specific).
    - Names that aren't valid Python identifiers are skipped with a warning
      (Pydantic field names must be identifiers; users who need exotic names
      should supply a full schema via --schema-path).
    - Mutates and returns schema_dict.
    """
    props = schema_dict.setdefault("properties", {})
    for name in hint_fields:
        if name in props:
            logger.info(
                f"[discovery] hint '{name}' already in discovered schema; "
                f"skipping injection (discovered field wins)"
            )
            continue
        if not _IDENT_RE.match(name):
            logger.warning(
                f"[discovery] hint '{name}' is not a valid Python identifier; "
                f"skipping (use --schema-path for non-identifier field names)"
            )
            continue
        # Use the anyOf form for Optional, matching what Pydantic emits for
        # Optional[str] fields. The list form `"type": ["string", "null"]` is
        # valid JSON Schema 2020 but synthesize_from_external_schema only
        # parses anyOf, so we'd otherwise round-trip-fail in the reconstructor.
        props[name] = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "title": name.replace("_", " ").title(),
            "description": (
                "User-hinted field (injected by --hint-fields). "
                "May be null if absent from the document."
            ),
            "default": None,
        }
        logger.info(f"[discovery] hint '{name}' injected as Optional[str]")
    return schema_dict


def run_discover_schema(
    pdf: Path,
    paths: StagePaths,
    *,
    extractor_model: str = DEFAULT_EXTRACTOR,
    schema_mode: str = "auto",
    schema_path: Optional[str] = None,
    hint_fields: Optional[List[str]] = None,
    force: bool = False,
) -> Optional[Tuple[DiscoveryResult, Type[BaseModel]]]:
    """Run the discovery stage.

    Returns (DiscoveryResult, response_model) when a fresh model was built so
    that in-process callers (notably stages.orchestrate.run_extract_group) can
    hand the Pydantic class straight to the extract stage without paying the
    JSON-Schema round-trip — the reconstructor at discovery_agent.py
    synthesize_from_external_schema is intentionally lossy and would otherwise
    drop nested types like SourceEvidence's typed page_number.

    Returns None on the smart-skip path (existing artifacts reused). Callers
    that don't need the in-memory model can ignore the return value entirely;
    standalone subprocess invocations still re-derive the model from the
    persisted auto_schema.json on the extract side.

    `hint_fields` (from --hint-fields) forcibly injects extra Optional[str]
    fields into the schema after discovery, so the extraction output always
    contains those keys (as null if the document doesn't supply them). When
    discovery artifacts already exist and one or more hints aren't in the
    on-disk schema, we warn and smart-skip anyway — pass --force to re-run.
    """
    paths.ensure()

    if paths.discovery.exists() and paths.auto_schema.exists() and not force:
        if hint_fields:
            existing = json.loads(paths.auto_schema.read_text(encoding="utf-8"))
            existing_props = set(existing.get("properties", {}).keys())
            missing = [h for h in hint_fields if h not in existing_props]
            if missing:
                logger.warning(
                    f"[discovery] hint fields {missing} not in existing "
                    f"{paths.auto_schema.name}; pass --force to re-discover "
                    f"with these hints"
                )
            else:
                logger.info(
                    f"[discovery] reusing existing schema (all hint fields "
                    f"already present)"
                )
            return None
        logger.info(
            f"[discovery] reusing existing {paths.discovery.name} + {paths.auto_schema.name}"
        )
        return None

    agent = DiscoveryAgent(model_id=extractor_model)

    if schema_mode == "explicit":
        if not schema_path:
            raise ValueError("schema_path is required when schema_mode='explicit'")
        logger.info(f"[discovery] explicit-schema mode (path: {schema_path})")
        response_model = agent.synthesize_from_external_schema(schema_path)
        domain = response_model.__name__
        # Mirror run_v3.py's dummy DiscoveryResult for the explicit path so the
        # downstream extract stage can rely on a consistent discovery.json shape.
        discovery_result = DiscoveryResult(
            domain=domain,
            is_high_density=True,
            dynamic_fields=[],
            confidence=1.0,
        )
    else:
        if not paths.markdown.exists():
            raise FileNotFoundError(
                f"Markdown artifact missing: {paths.markdown}. Run pdf-to-markdown first."
            )
        markdown = paths.markdown.read_text(encoding="utf-8")
        doc_preview = markdown[:10000]
        graph_summary = (
            paths.graph_summary.read_text(encoding="utf-8")
            if paths.graph_summary.exists()
            else ""
        )

        discovery_result = agent.scout(doc_preview, graph_summary=graph_summary)
        response_model = agent.synthesize_model(discovery_result)
        domain = discovery_result.domain

    schema_dict = response_model.model_json_schema()
    if hint_fields:
        schema_dict = _inject_hint_fields(schema_dict, hint_fields)

    paths.discovery.write_text(
        json.dumps(discovery_result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.auto_schema.write_text(
        json.dumps(schema_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if hint_fields:
        # Re-synthesize the model from the modified on-disk schema so the
        # in-process caller (run_extract_group) gets a Pydantic model that
        # includes the hint fields. The synthesize path preserves $ref/anyOf.
        response_model = agent.synthesize_from_external_schema(str(paths.auto_schema))

    logger.info(
        f"[discovery] domain={domain}, "
        f"is_high_density={discovery_result.is_high_density} → "
        f"{paths.discovery.name} + {paths.auto_schema.name}"
    )
    return discovery_result, response_model
