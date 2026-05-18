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
from pathlib import Path
from typing import Optional

from extractor.discovery_agent import DiscoveryAgent, DiscoveryResult
from stages.paths import StagePaths

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTOR = "RMunshi/librarian-qwen-extractor"


def run_discover_schema(
    pdf: Path,
    paths: StagePaths,
    *,
    extractor_model: str = DEFAULT_EXTRACTOR,
    schema_mode: str = "auto",
    schema_path: Optional[str] = None,
    force: bool = False,
) -> None:
    paths.ensure()

    if paths.discovery.exists() and paths.auto_schema.exists() and not force:
        logger.info(
            f"[discovery] reusing existing {paths.discovery.name} + {paths.auto_schema.name}"
        )
        return

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

    paths.discovery.write_text(
        json.dumps(discovery_result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.auto_schema.write_text(
        json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"[discovery] domain={domain}, "
        f"is_high_density={discovery_result.is_high_density} → "
        f"{paths.discovery.name} + {paths.auto_schema.name}"
    )
