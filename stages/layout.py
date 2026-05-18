"""
Stage 1 — pdf-to-layout.

Reads:   <pdf>
Writes:  <doc_stem>_regions.json  (List[LayoutRegion])
         + the EnhancedPipeline's incidental artifacts (enhanced_layout_blocks.json,
           layout_thumbnails/, layout_visualizations.pdf) into output_dir.

Cost: loads the 22 GB Vision model. This is the heaviest single stage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from converter.engine import ConverterEngine
from stages.paths import StagePaths
from stages.serialize import save_regions

logger = logging.getLogger(__name__)

DEFAULT_VLM = "RMunshi/vlm-student-thesis"


def run_pdf_to_layout(
    pdf: Path,
    paths: StagePaths,
    *,
    vlm_model: str = DEFAULT_VLM,
    vlm_provider: Optional[str] = None,
    max_pages: Optional[int] = None,
    debug: bool = False,
    force: bool = False,
) -> None:
    paths.ensure()

    if paths.regions.exists() and not force:
        logger.info(f"[layout] reusing existing {paths.regions.name} (pass force=True to rerun)")
        return

    logger.info(f"[layout] {pdf.name} → {paths.regions.name} (VLM: {vlm_model})")
    converter = ConverterEngine(
        output_dir=str(paths.output_dir),
        debug=debug,
        vlm_model=vlm_model,
        vlm_provider=vlm_provider,
    )
    regions = converter.convert_to_regions(str(pdf), max_pages=max_pages)
    save_regions(regions, paths.regions)
    logger.info(f"[layout] wrote {len(regions)} regions across {len({r.page for r in regions})} pages")
