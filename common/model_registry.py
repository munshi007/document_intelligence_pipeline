"""
Model asset registry — reads ``config/models.yaml`` and serves pinned
(repo_id, revision, filename) tuples to every model-loading site in the
pipeline.

Why this exists: thesis-grade reproducibility requires that each HF asset
be fetched at the exact commit SHA used to produce the published numbers.
Centralising the pins here means a reviewer can audit the manifest in one
place rather than chasing inline strings across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    kind: str
    filename: Optional[str] = None
    role: str = ""
    approx_size_gb: float = 0.0
    license: str = ""


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "models.yaml"


@lru_cache(maxsize=1)
def _load_manifest() -> Dict[str, ModelSpec]:
    path = _manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model manifest not found at {path}. "
            "Run `python scripts/bootstrap.py` to verify your checkout."
        )
    raw = yaml.safe_load(path.read_text())
    specs: Dict[str, ModelSpec] = {}
    for entry in raw.get("models", []):
        spec = ModelSpec(
            name=entry["name"],
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            kind=entry["kind"],
            filename=entry.get("filename"),
            role=entry.get("role", ""),
            approx_size_gb=float(entry.get("approx_size_gb", 0.0)),
            license=entry.get("license", ""),
        )
        specs[spec.name] = spec
    return specs


def get(name: str) -> ModelSpec:
    """Return the pinned ModelSpec for ``name``. Raises KeyError if missing."""
    manifest = _load_manifest()
    if name not in manifest:
        raise KeyError(
            f"Model '{name}' not in manifest. "
            f"Known: {sorted(manifest.keys())}"
        )
    return manifest[name]


def all_specs() -> Dict[str, ModelSpec]:
    """Return the full manifest. Used by bootstrap.py for preflight downloads."""
    return dict(_load_manifest())
