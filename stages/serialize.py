"""
Disk handoff for LayoutRegion lists.

Stage 1 (pdf-to-layout) produces a List[LayoutRegion] that stages 2 (markdown)
and 3 (graph) consume. When stages run in separate subprocesses, that list
crosses a process boundary — so we serialize it via Pydantic model_dump,
preserving every field including nested metadata.

The graph itself already has dedicated save_graph / load_graph helpers in
storage/store.py, and the discovery output is reconstructed from auto_schema
via DiscoveryAgent.synthesize_from_external_schema — so this module only has
to handle the regions handoff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from core.schemas import LayoutRegion


def save_regions(regions: List[LayoutRegion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in regions]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_regions(path: Path) -> List[LayoutRegion]:
    if not path.exists():
        raise FileNotFoundError(
            f"Regions artifact missing: {path}\n"
            f"Run `cli pdf-to-layout` first, or invoke run-all."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [LayoutRegion(**d) for d in data]
