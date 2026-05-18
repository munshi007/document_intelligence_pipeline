"""
Stage groups — multi-stage helpers that share one Python process.

Running each stage as its own OS subprocess would mean reloading the Vision
model (22 GB) for layout and the Text model (15 GB) for discovery+extract.
That's wasteful. The natural groupings are:

    layout_group  = pdf-to-layout → pdf-to-markdown → md-to-graph
                    (the Vision model loads once for layout; markdown and
                    graph are pure-Python and add no model cost)

    extract_group = discover-schema → extract
                    (the Text model loads once in DiscoveryAgent and is
                    reused by ExtractorAgent)

cli.py's `run-all` subcommand subprocess-chains these two groups, which is
what gives us the GPU memory boundary "for free" (the layout subprocess
exits before the extract subprocess starts).

These helpers are also useful for ad-hoc scripts that want a half-pipeline
without going through the CLI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from stages.discovery import DEFAULT_EXTRACTOR, run_discover_schema
from stages.extract import run_extract
from stages.graph import run_md_to_graph
from stages.layout import DEFAULT_VLM, run_pdf_to_layout
from stages.markdown import run_pdf_to_markdown
from stages.paths import StagePaths

logger = logging.getLogger(__name__)


def run_layout_group(
    pdf: Path,
    paths: StagePaths,
    *,
    vlm_model: str = DEFAULT_VLM,
    vlm_provider: Optional[str] = None,
    max_pages: Optional[int] = None,
    debug: bool = False,
    force: bool = False,
) -> None:
    """Layout → markdown → graph in one process."""
    logger.info(f"[group:layout] starting on {pdf.name}")
    run_pdf_to_layout(
        pdf, paths,
        vlm_model=vlm_model, vlm_provider=vlm_provider,
        max_pages=max_pages, debug=debug, force=force,
    )
    run_pdf_to_markdown(pdf, paths, debug=debug, force=force)
    run_md_to_graph(pdf, paths, force=force)
    logger.info(f"[group:layout] done")


def run_extract_group(
    pdf: Path,
    paths: StagePaths,
    *,
    extractor_model: str = DEFAULT_EXTRACTOR,
    schema_mode: str = "auto",
    schema_path: Optional[str] = None,
    with_grounding: bool = False,
    save_debug_traces: bool = False,
    distill: bool = False,
    force: bool = False,
) -> None:
    """Discover-schema → extract in one process (text model loaded once)."""
    logger.info(f"[group:extract] starting on {pdf.name}")
    run_discover_schema(
        pdf, paths,
        extractor_model=extractor_model,
        schema_mode=schema_mode,
        schema_path=schema_path,
        force=force,
    )
    run_extract(
        pdf, paths,
        extractor_model=extractor_model,
        with_grounding=with_grounding,
        save_debug_traces=save_debug_traces,
        distill=distill,
        force=force,
    )
    logger.info(f"[group:extract] done")
