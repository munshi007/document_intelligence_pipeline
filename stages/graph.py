"""
Stage 3 — md-to-graph.

Reads:   <doc_stem>_regions.json
Writes:  storage/<doc_stem>_graph.json  (via EphemeralStore.save_graph)
         <doc_stem>_graph_summary.txt   (human-readable companion)

Cost: pure Python state machine. No models. Cheap to re-run.
"""
from __future__ import annotations

import logging
from pathlib import Path

from chunker.graph_builder import GraphBuilder
from storage.store import EphemeralStore
from stages.paths import StagePaths
from stages.serialize import load_regions

logger = logging.getLogger(__name__)


def run_md_to_graph(
    pdf: Path,
    paths: StagePaths,
    *,
    force: bool = False,
) -> None:
    paths.ensure()

    if paths.graph_json.exists() and paths.graph_summary.exists() and not force:
        logger.info(f"[graph] reusing existing {paths.graph_json.name} + {paths.graph_summary.name}")
        return

    regions = load_regions(paths.regions)
    doc_info = {
        "doc_id": paths.doc_stem,
        "filename": pdf.name,
        "total_pages": len({r.page for r in regions}),
    }

    chunker = GraphBuilder()
    graph = chunker.build_graph(regions, doc_info)

    store = EphemeralStore(storage_dir=str(paths.graph_storage_dir))
    store.save_graph(graph)

    with open(paths.graph_summary, "w", encoding="utf-8") as f:
        f.write(f"Document Graph Summary: {doc_info['filename']}\n")
        f.write("=" * 80 + "\n")
        for node in graph.nodes:
            f.write(f"[{node.node_id}] {node.metadata.get('breadcrumb', 'Root')}\n")
            f.write(f"   Content: {node.content[:200]}...\n\n")

    logger.info(f"[graph] wrote {len(graph.nodes)} nodes → {paths.graph_json.name}")
