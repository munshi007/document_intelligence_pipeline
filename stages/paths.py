"""
Canonical artifact paths for the pipeline.

Every filename produced by every stage funnels through one of these properties,
so we never have to chase "what was that file called again?" across stages.
The CLI, individual stages, and run_v3.py all import from here.

Filename conventions mirror what run_v3.py historically wrote, so existing
output directories (e.g. output/v3) and downstream tooling (eval scripts,
README references) keep working unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

# Longest filename suffix any stage writes is ``_universal_extraction.json``
# (26 chars), so doc_stem must stay under 255 - 26 = 229 to be safe. We
# pick 200 as a comfortable headroom for any future suffix growth and to
# leave the filename humane-looking when truncated.
_MAX_STEM_LEN = 200


def safe_stem(stem: str) -> str:
    """Truncate excessively long PDF stems to fit OS filename limits.

    Some upstream PDFs ship with stems north of 250 chars (URL-encoded
    Google Drive exports, scanner output, etc.) which overflows the
    255-char filename limit once any of our suffixes (``_universal_extraction.json``)
    is appended. We deterministically hash-suffix anything past ``_MAX_STEM_LEN``
    so the same PDF always maps to the same stem on disk, and so the truncated
    head remains a useful breadcrumb when grepping through ``output/``.
    """
    if len(stem) <= _MAX_STEM_LEN:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:40]}__{digest}"


@dataclass(frozen=True)
class StagePaths:
    """Resolves every artifact path for a single (output_dir, doc_stem)."""

    output_dir: Path
    doc_stem: str

    @classmethod
    def for_pdf(cls, pdf: PathLike, output_dir: PathLike) -> "StagePaths":
        return cls(output_dir=Path(output_dir), doc_stem=safe_stem(Path(pdf).stem))

    def ensure(self) -> None:
        """Create output and storage dirs. Cheap, idempotent, side-effect only."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

    # ── stage 1: pdf-to-layout ───────────────────────────────────────────────
    @property
    def regions(self) -> Path:
        """List[LayoutRegion] serialized as JSON (new — for cross-process handoff)."""
        return self.output_dir / f"{self.doc_stem}_regions.json"

    # ── stage 2: pdf-to-markdown ─────────────────────────────────────────────
    @property
    def markdown(self) -> Path:
        return self.output_dir / "extracted_content.md"

    @property
    def manifest(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_manifest.json"

    # ── stage 3: md-to-graph ─────────────────────────────────────────────────
    @property
    def graph_summary(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_graph_summary.txt"

    @property
    def graph_storage_dir(self) -> Path:
        return self.output_dir / "storage"

    @property
    def graph_json(self) -> Path:
        """Where EphemeralStore.save_graph writes the full graph."""
        return self.graph_storage_dir / f"{self.doc_stem}_graph.json"

    # ── stage 4: discover-schema ─────────────────────────────────────────────
    @property
    def discovery(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_discovery.json"

    @property
    def auto_schema(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_auto_schema.json"

    # ── stage 5: extract ─────────────────────────────────────────────────────
    @property
    def extraction(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_universal_extraction.json"

    @property
    def grounding(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_grounding.json"

    @property
    def evaluation(self) -> Path:
        return self.output_dir / f"{self.doc_stem}_eval.json"

    @property
    def debug_traces(self) -> Path:
        return self.output_dir / "debug_traces"
