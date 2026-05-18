"""
Document Intelligence Pipeline — per-stage CLI.

Each subcommand maps to one stage in stages/, with smart-skip (existing
output is reused unless --force is passed) and auto-chain (missing inputs
trigger the prior stage automatically). Composite subcommands and the
top-level `run-all` orchestrator subprocess-chain the natural stage groups
so the Vision and Text models never share VRAM in the same Python process.

Usage:
    python -m cli run-all data/sample.pdf
    python -m cli run-all data/sample.pdf --extract --with-grounding
    python -m cli pdf-to-layout data/sample.pdf
    python -m cli pdf-to-markdown data/sample.pdf       # auto-runs layout if needed
    python -m cli md-to-graph data/sample.pdf
    python -m cli discover-schema data/sample.pdf
    python -m cli extract data/sample.pdf
    python -m cli pdf-to-graph data/sample.pdf          # composite: layout+md+graph
    python -m cli discover-and-extract data/sample.pdf  # composite: discovery+extract
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

# Quiet the noisy libs the same way run_v3.py does, before any heavy import.
logging.getLogger("unsloth").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("trl").setLevel(logging.CRITICAL)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

# Make sibling packages importable when cli.py is run via `python -m cli` or
# directly. The repo root is the file's parent.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stages.discovery import DEFAULT_EXTRACTOR, run_discover_schema
from stages.extract import run_extract
from stages.graph import run_md_to_graph
from stages.layout import DEFAULT_VLM, run_pdf_to_layout
from stages.markdown import run_pdf_to_markdown
from stages.orchestrate import run_extract_group, run_layout_group
from stages.paths import StagePaths


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run the document-intelligence pipeline one stage at a time, or end-to-end.",
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ────────────────────────── auto-chain helpers ──────────────────────────
# Each _ensure_* helper guarantees its stage's primary outputs exist by
# running prior stages in-process if necessary. Used by individual stage
# subcommands so a user can call any stage and have the prerequisites
# satisfied transparently.

def _ensure_layout(pdf: Path, paths: StagePaths, *, vlm_model: str, max_pages: Optional[int], debug: bool) -> None:
    if paths.regions.exists():
        return
    run_pdf_to_layout(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)


def _ensure_markdown(pdf: Path, paths: StagePaths, *, vlm_model: str, max_pages: Optional[int], debug: bool) -> None:
    if paths.markdown.exists() and paths.manifest.exists():
        return
    _ensure_layout(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_pdf_to_markdown(pdf, paths, debug=debug)


def _ensure_graph(pdf: Path, paths: StagePaths, *, vlm_model: str, max_pages: Optional[int], debug: bool) -> None:
    if paths.graph_json.exists() and paths.graph_summary.exists():
        return
    _ensure_layout(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_md_to_graph(pdf, paths)


def _ensure_discovery(
    pdf: Path,
    paths: StagePaths,
    *,
    vlm_model: str,
    max_pages: Optional[int],
    debug: bool,
    extractor_model: str,
    schema_mode: str,
    schema_path: Optional[str],
) -> None:
    if paths.discovery.exists() and paths.auto_schema.exists():
        return
    _ensure_markdown(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    _ensure_graph(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_discover_schema(
        pdf, paths,
        extractor_model=extractor_model,
        schema_mode=schema_mode,
        schema_path=schema_path,
    )


# ────────────────────────── atomic stage commands ───────────────────────

@app.command("pdf-to-layout")
def cmd_pdf_to_layout(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stage 1: extract layout regions from a PDF (loads the Vision model)."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    run_pdf_to_layout(pdf, paths,
                      vlm_model=vlm_model, vlm_provider=vlm_provider,
                      max_pages=max_pages, debug=debug, force=force)


@app.command("pdf-to-markdown")
def cmd_pdf_to_markdown(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stage 2: build structured Markdown + manifest from layout regions."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    _ensure_layout(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_pdf_to_markdown(pdf, paths, debug=debug, force=force)


@app.command("md-to-graph")
def cmd_md_to_graph(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stage 3: build the Hierarchical Knowledge Graph from layout regions."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    _ensure_layout(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_md_to_graph(pdf, paths, force=force)


@app.command("discover-schema")
def cmd_discover_schema(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    extractor_model: str = typer.Option(DEFAULT_EXTRACTOR, "--extractor-model"),
    schema_mode: str = typer.Option("auto", "--schema-mode",
                                    help="auto | explicit"),
    schema_path: Optional[str] = typer.Option(None, "--schema-path",
                                              help="JSON schema path for --schema-mode explicit"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stage 4: discover the extraction schema (loads the Text model)."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    # Discovery needs markdown + graph_summary, which need layout first.
    _ensure_markdown(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    _ensure_graph(pdf, paths, vlm_model=vlm_model, max_pages=max_pages, debug=debug)
    run_discover_schema(
        pdf, paths,
        extractor_model=extractor_model,
        schema_mode=schema_mode,
        schema_path=schema_path,
        force=force,
    )


@app.command("extract")
def cmd_extract(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    extractor_model: str = typer.Option(DEFAULT_EXTRACTOR, "--extractor-model"),
    schema_mode: str = typer.Option("auto", "--schema-mode",
                                    help="auto | explicit"),
    schema_path: Optional[str] = typer.Option(None, "--schema-path",
                                              help="JSON schema path for --schema-mode explicit"),
    with_grounding: bool = typer.Option(False, "--with-grounding"),
    save_debug_traces: bool = typer.Option(False, "--save-debug-traces"),
    distill: bool = typer.Option(False, "--distill"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stage 5: run extraction against the discovered/explicit schema."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    _ensure_discovery(
        pdf, paths,
        vlm_model=vlm_model, max_pages=max_pages, debug=debug,
        extractor_model=extractor_model,
        schema_mode=schema_mode, schema_path=schema_path,
    )
    run_extract(
        pdf, paths,
        extractor_model=extractor_model,
        with_grounding=with_grounding,
        save_debug_traces=save_debug_traces,
        distill=distill,
        force=force,
    )


# ────────────────────────── composite subprocess groups ─────────────────
# These are what run-all subprocess-chains. They're also useful on their
# own: pdf-to-graph builds every "input" artifact a downstream tool needs
# without ever loading the Text model, and discover-and-extract picks up
# where pdf-to-graph leaves off.

@app.command("pdf-to-graph")
def cmd_pdf_to_graph(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Composite: layout + markdown + graph in one process (Vision loads once)."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    run_layout_group(
        pdf, paths,
        vlm_model=vlm_model, vlm_provider=vlm_provider,
        max_pages=max_pages, debug=debug, force=force,
    )


@app.command("discover-and-extract")
def cmd_discover_and_extract(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    extractor_model: str = typer.Option(DEFAULT_EXTRACTOR, "--extractor-model"),
    schema_mode: str = typer.Option("auto", "--schema-mode"),
    schema_path: Optional[str] = typer.Option(None, "--schema-path"),
    with_grounding: bool = typer.Option(False, "--with-grounding"),
    save_debug_traces: bool = typer.Option(False, "--save-debug-traces"),
    distill: bool = typer.Option(False, "--distill"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Composite: discovery + extract in one process (Text model loads once)."""
    _setup_logging()
    paths = StagePaths.for_pdf(pdf, output_dir)
    run_extract_group(
        pdf, paths,
        extractor_model=extractor_model,
        schema_mode=schema_mode, schema_path=schema_path,
        with_grounding=with_grounding,
        save_debug_traces=save_debug_traces,
        distill=distill,
        force=force,
    )


# ────────────────────────── run-all orchestrator ────────────────────────

@app.command("run-all")
def cmd_run_all(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output/v3"), "--output-dir", "-o"),
    extract_flag: bool = typer.Option(False, "--extract",
                                      help="Also run discovery + extraction."),
    vlm_model: str = typer.Option(DEFAULT_VLM, "--vlm-model"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider"),
    extractor_model: str = typer.Option(DEFAULT_EXTRACTOR, "--extractor-model"),
    schema_mode: str = typer.Option("auto", "--schema-mode"),
    schema_path: Optional[str] = typer.Option(None, "--schema-path"),
    with_grounding: bool = typer.Option(False, "--with-grounding"),
    save_debug_traces: bool = typer.Option(False, "--save-debug-traces"),
    distill: bool = typer.Option(False, "--distill"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    debug: bool = typer.Option(False, "--debug"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the whole pipeline as two subprocess-chained stage groups."""
    _setup_logging()
    logger = logging.getLogger("cli.run-all")

    # Group 1 — layout/markdown/graph. Vision model loads inside this subprocess
    # and is released cleanly when the process exits.
    group1: list[str] = [
        sys.executable, "-m", "cli", "pdf-to-graph", str(pdf),
        "--output-dir", str(output_dir),
        "--vlm-model", vlm_model,
    ]
    if vlm_provider:    group1 += ["--vlm-provider", vlm_provider]
    if max_pages:       group1 += ["--max-pages", str(max_pages)]
    if debug:           group1.append("--debug")
    if force:           group1.append("--force")

    logger.info(f"[run-all] Group 1 (layout/markdown/graph): {' '.join(group1)}")
    r1 = subprocess.run(group1, cwd=str(_REPO_ROOT))
    if r1.returncode != 0:
        logger.error(f"[run-all] Group 1 failed (exit {r1.returncode})")
        raise typer.Exit(code=r1.returncode)

    if not extract_flag:
        logger.info("[run-all] --extract not set; stopping after Group 1")
        return

    # Group 2 — discovery + extract. Text model loads fresh in a new process,
    # so the layout subprocess's VRAM doesn't compete.
    group2: list[str] = [
        sys.executable, "-m", "cli", "discover-and-extract", str(pdf),
        "--output-dir", str(output_dir),
        "--extractor-model", extractor_model,
        "--schema-mode", schema_mode,
    ]
    if schema_path:         group2 += ["--schema-path", schema_path]
    if with_grounding:      group2.append("--with-grounding")
    if save_debug_traces:   group2.append("--save-debug-traces")
    if distill:             group2.append("--distill")
    if force:               group2.append("--force")

    logger.info(f"[run-all] Group 2 (discover/extract): {' '.join(group2)}")
    r2 = subprocess.run(group2, cwd=str(_REPO_ROOT))
    if r2.returncode != 0:
        logger.error(f"[run-all] Group 2 failed (exit {r2.returncode})")
        raise typer.Exit(code=r2.returncode)

    logger.info("[run-all] pipeline complete")


if __name__ == "__main__":
    app()
