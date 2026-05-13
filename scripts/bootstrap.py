"""
Bootstrap — one-shot reproducibility verifier.

What it does, in order:
  1. Preflight    — Python version, CUDA availability, free disk, manifest files.
  2. Model fetch  — downloads every entry in config/models.yaml at its pinned
                    revision SHA into the HuggingFace cache.
  3. Smoke test   — runs the pipeline end-to-end on the committed sample PDFs
                    (data/simple_invoice.pdf, data/Super_Complex_2.pdf) and
                    checks that each produces a non-empty extraction.
  4. Sentinel     — writes BOOTSTRAP_OK with a JSON summary that an examiner
                    can paste into a defence appendix.

Usage:
    python scripts/bootstrap.py                  # full run
    python scripts/bootstrap.py --skip-download  # rerun smoke only
    python scripts/bootstrap.py --skip-smoke     # download only
    python scripts/bootstrap.py --yes            # skip the size confirmation
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDFS = [
    PROJECT_ROOT / "data" / "simple_invoice.pdf",
    PROJECT_ROOT / "data" / "Super_Complex_2.pdf",
]
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "output" / "bootstrap_smoke"
SENTINEL_PATH = PROJECT_ROOT / "BOOTSTRAP_OK"


# ────────────────────────────── pretty-print helpers ──────────────────────────

def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {msg}\n{bar}")


def step(msg: str) -> None:
    print(f"\n[*] {msg}")


def ok(msg: str) -> None:
    print(f"    [OK] {msg}")


def warn(msg: str) -> None:
    print(f"    [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"    [FAIL] {msg}", file=sys.stderr)


# ────────────────────────────── preflight checks ──────────────────────────────

def check_python() -> None:
    if sys.version_info < (3, 12):
        fail(f"Python 3.12+ required (current: {sys.version.split()[0]})")
        sys.exit(2)
    ok(f"Python {sys.version.split()[0]}")


def check_cuda() -> None:
    try:
        import torch
    except Exception as e:
        fail(f"Cannot import torch ({e}). Did `conda env create -f environment.yml` succeed?")
        sys.exit(2)
    if not torch.cuda.is_available():
        warn("CUDA not available — pipeline will run on CPU (very slow).")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    ok(f"CUDA available — {name} (compute {cap[0]}.{cap[1]}), torch={torch.__version__}")


def check_manifest_files() -> None:
    for p in (
        PROJECT_ROOT / "environment.yml",
        PROJECT_ROOT / "config" / "models.yaml",
    ):
        if not p.exists():
            fail(f"Missing manifest file: {p}")
            sys.exit(2)
    ok("environment.yml + config/models.yaml present")


def check_sample_pdfs() -> None:
    missing = [p for p in SAMPLE_PDFS if not p.exists()]
    if missing:
        fail("Sample PDFs not in repo: " + ", ".join(str(p) for p in missing))
        sys.exit(2)
    ok(f"Sample PDFs present ({len(SAMPLE_PDFS)})")


def check_free_disk(required_gb: float) -> None:
    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache_root).free / (1024 ** 3)
    if free_gb < required_gb + 5:  # 5 GB headroom
        fail(f"Free disk at {cache_root}: {free_gb:.1f} GB (need ~{required_gb:.1f} GB + 5 GB headroom)")
        sys.exit(2)
    ok(f"Free disk at {cache_root}: {free_gb:.1f} GB (need ~{required_gb:.1f} GB)")


# ────────────────────────────── model fetch ──────────────────────────────────

def fetch_models(assume_yes: bool) -> dict:
    from common import model_registry

    specs = model_registry.all_specs()
    total_gb = sum(s.approx_size_gb for s in specs.values())

    step(f"Preparing to fetch {len(specs)} model(s), total ~{total_gb:.1f} GB")
    for s in specs.values():
        print(f"      - {s.name:20s} {s.repo_id}@{s.revision[:10]}  ({s.approx_size_gb:.1f} GB)  {s.license}")

    if not assume_yes:
        reply = input("\n    Proceed with download? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("    Aborted by user.")
            sys.exit(1)

    from huggingface_hub import hf_hub_download, snapshot_download

    summary = {}
    for s in specs.values():
        t0 = time.time()
        try:
            if s.filename:
                path = hf_hub_download(repo_id=s.repo_id, filename=s.filename, revision=s.revision)
            else:
                path = snapshot_download(repo_id=s.repo_id, revision=s.revision)
        except Exception as e:
            fail(f"Download failed for {s.name}: {e}")
            sys.exit(3)
        elapsed = time.time() - t0
        ok(f"{s.name} → {path}  (took {elapsed:.1f}s)")
        summary[s.name] = {"repo_id": s.repo_id, "revision": s.revision, "path": str(path)}
    return summary


# ────────────────────────────── smoke test ───────────────────────────────────

def run_smoke(pdf_path: Path) -> dict:
    SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "run_v3.py", str(pdf_path),
        "--output_dir", str(SMOKE_OUTPUT_DIR),
        "--extract", "--schema_mode", "auto",
    ]
    print(f"      $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    stem = pdf_path.stem
    scorecard_path = SMOKE_OUTPUT_DIR / f"{stem}_universal_extraction.json"
    payload = {"pdf": str(pdf_path.name), "exit_code": result.returncode, "elapsed_s": round(elapsed, 1)}
    if result.returncode != 0:
        payload["stderr_tail"] = result.stderr[-2000:]
        fail(f"Pipeline failed for {pdf_path.name} (exit {result.returncode})")
        return payload
    if not scorecard_path.exists():
        payload["error"] = f"Expected output {scorecard_path.name} not produced"
        fail(payload["error"])
        return payload
    try:
        data = json.loads(scorecard_path.read_text())
        non_empty = sum(1 for v in data.values() if v not in (None, "", [], {}))
        payload["fields_non_empty"] = non_empty
        ok(f"{pdf_path.name}: {non_empty} non-empty top-level fields, {elapsed:.1f}s")
    except Exception as e:
        payload["error"] = f"Could not parse output JSON: {e}"
        fail(payload["error"])
    return payload


# ────────────────────────────── main flow ────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducibility bootstrap")
    parser.add_argument("--skip-download", action="store_true", help="Skip the model-fetch stage.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the smoke-test stage.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip download confirmation prompt.")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    banner("STAGE 1: Preflight")
    check_python()
    check_manifest_files()
    check_sample_pdfs()
    check_cuda()

    from common import model_registry
    total_gb = sum(s.approx_size_gb for s in model_registry.all_specs().values())
    check_free_disk(total_gb)

    sentinel = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "stages": {},
    }

    if not args.skip_download:
        banner("STAGE 2: Model fetch")
        sentinel["stages"]["models"] = fetch_models(assume_yes=args.yes)
    else:
        warn("STAGE 2 skipped (--skip-download)")

    if not args.skip_smoke:
        banner("STAGE 3: Smoke test")
        smoke_results = []
        for pdf in SAMPLE_PDFS:
            step(f"Running pipeline on {pdf.name}")
            smoke_results.append(run_smoke(pdf))
        sentinel["stages"]["smoke"] = smoke_results
        if any(r.get("exit_code", 1) != 0 for r in smoke_results):
            fail("One or more smoke tests failed — see stderr_tail in BOOTSTRAP_OK")
            SENTINEL_PATH.write_text(json.dumps(sentinel, indent=2))
            return 4
    else:
        warn("STAGE 3 skipped (--skip-smoke)")

    sentinel["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    SENTINEL_PATH.write_text(json.dumps(sentinel, indent=2))
    banner(f"BOOTSTRAP_OK — wrote {SENTINEL_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
