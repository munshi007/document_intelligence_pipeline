"""
run_v3.py — back-compat shim for the per-stage CLI.

Preserves the exact command-line interface that scripts/bootstrap.py, README
examples, and any wrapper scripts already use. Internally it just translates
the flags into the equivalent `python -m cli run-all ...` invocation, so the
manual VRAM-flush hack that used to live here is no longer needed — the
subprocess boundary between the layout and extract groups handles GPU memory
isolation automatically.

If you're writing new code, prefer `python -m cli <stage> ...` directly. It
exposes the individual stages with smart-skip + auto-chain and uses
Typer-standard dashed flag names.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Flags that the historical run_v3.py declared but never read. We still accept
# them so existing wrapper scripts don't crash, but we warn the user that the
# flag has no effect.
_DEAD_FLAGS = (
    "--auto_schema",
    "--evaluate",
    "--no_routing",
    "--no_projection",
    "--with_normalization",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Document Intelligence Pipeline (back-compat shim — calls cli.py run-all internally)"
    )
    p.add_argument("pdf_path", type=str, help="Path to the source PDF file")
    p.add_argument("--output_dir", type=str, default="output/v3", help="Base output directory")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    p.add_argument("--max_pages", type=int, help="Limit number of pages")

    # Extraction options
    p.add_argument("--extract", action="store_true", help="Enable extraction step")
    p.add_argument("--auto_schema", action="store_true",
                   help="(Deprecated, no-op) Discovery is now the default — use --schema_mode auto|explicit.")
    p.add_argument("--schema_mode", type=str, choices=["auto", "domain", "explicit"],
                   default="auto", help="Schema routing mode (domain is treated as auto).")
    p.add_argument("--schema_path", type=str, default=None,
                   help="Path to explicit extraction schema JSON (used with --schema_mode explicit)")
    p.add_argument("--save_debug_traces", action="store_true",
                   help="Create debug trace directory for extraction diagnostics")
    p.add_argument("--evaluate", action="store_true",
                   help="(No-op) Evaluation now always runs when --extract is set.")
    p.add_argument("--extractor_model", type=str,
                   default="RMunshi/librarian-qwen-extractor",
                   help="Text model for extraction")
    p.add_argument("--model", type=str,
                   default="RMunshi/vlm-student-thesis",
                   help="Vision model for layout parsing")
    p.add_argument("--distill", action="store_true", help="Enable distillation (data capture)")
    p.add_argument("--with_grounding", action="store_true",
                   help="Enable langextract-based precision grounding")

    # Ablation flags — historically declared but never wired through the pipeline.
    p.add_argument("--no_routing", action="store_true",
                   help="(No-op) Ablation A0 placeholder; kept for back-compat.")
    p.add_argument("--no_projection", action="store_true",
                   help="(No-op) Ablation A0/A1 placeholder; kept for back-compat.")
    p.add_argument("--with_normalization", action="store_true",
                   help="(No-op) Ablation A3 placeholder; kept for back-compat.")
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logger = logging.getLogger("run_v3")

    args = _build_parser().parse_args()

    # Warn about dead flags actually set on this invocation.
    for name in _DEAD_FLAGS:
        attr = name.lstrip("-")
        if getattr(args, attr, False):
            logger.warning(f"{name} is accepted for back-compat but has no effect.")

    # 'domain' historically routed to the discovery path; the new CLI's auto
    # mode is the same code path, so collapse them.
    schema_mode = "auto" if args.schema_mode == "domain" else args.schema_mode

    cmd: list[str] = [
        sys.executable, "-m", "cli", "run-all", args.pdf_path,
        "--output-dir", args.output_dir,
        "--vlm-model", args.model,
        "--extractor-model", args.extractor_model,
        "--schema-mode", schema_mode,
    ]
    if args.extract:           cmd.append("--extract")
    if args.with_grounding:    cmd.append("--with-grounding")
    if args.save_debug_traces: cmd.append("--save-debug-traces")
    if args.distill:           cmd.append("--distill")
    if args.debug:             cmd.append("--debug")
    if args.max_pages:         cmd += ["--max-pages", str(args.max_pages)]
    if args.schema_path:       cmd += ["--schema-path", args.schema_path]

    logger.info(f"Delegating to: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
