"""
Stage 2 — pdf-to-markdown.

Reads:   <doc_stem>_regions.json
Writes:  extracted_content.md, <doc_stem>_manifest.json

Cost: pure Python over the regions list, no model load. Free with the lazy-
init ConverterEngine (constructing the engine only loads the VLM when
self.pipeline is actually accessed, which build_markdown_and_manifest never
does).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from converter.engine import ConverterEngine
from stages.paths import StagePaths
from stages.serialize import load_regions

logger = logging.getLogger(__name__)


def run_pdf_to_markdown(
    pdf: Path,
    paths: StagePaths,
    *,
    debug: bool = False,
    force: bool = False,
) -> None:
    paths.ensure()

    if paths.markdown.exists() and paths.manifest.exists() and not force:
        logger.info(f"[markdown] reusing existing {paths.markdown.name} + {paths.manifest.name}")
        return

    regions = load_regions(paths.regions)
    doc_info = {
        "doc_id": paths.doc_stem,
        "filename": pdf.name,
        "total_pages": len({r.page for r in regions}),
    }

    # Construct only to call build_markdown_and_manifest — pipeline stays unloaded.
    converter = ConverterEngine(output_dir=str(paths.output_dir), debug=debug)
    markdown, manifest = converter.build_markdown_and_manifest(regions, doc_info)

    paths.markdown.write_text(markdown, encoding="utf-8")
    paths.manifest.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"[markdown] wrote {paths.markdown.name} ({len(markdown)} chars) + {paths.manifest.name}")
