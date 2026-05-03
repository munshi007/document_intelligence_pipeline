"""
Provenance-Preserving Field Normalizer
=======================================
Every extracted scalar is wrapped in a FieldValue envelope that carries:
  - raw_value            : the string exactly as it appeared in the document
  - normalized_value     : the cleaned, typed value after rule application
  - normalization_rule   : name of the rule that was applied (or None)
  - normalization_confidence : float in [0,1]; below CONFIDENCE_GATE the
                               normalizer keeps raw_value and sets unresolved=True
  - source_node_id       : graph node where the value was found (optional)
  - source_page          : 1-based page number (optional)
  - grounded             : True when source_node_id is set
  - unresolved           : True when confidence < CONFIDENCE_GATE; the consumer
                           should fall back to raw_value in that case

Ablation-safe design
---------------------
  normalizer = FieldNormalizer()
  fv = normalizer.normalize(raw, field_type_hint="amount")

All rules are named constants so callers can filter results by rule in an
ablation condition (e.g.  results where normalization_rule IS NOT None).

Rules are independent of extraction logic — they only operate on strings.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_GATE: float = 0.65
"""Below this threshold the normalizer marks the value as unresolved and
leaves normalized_value identical to raw_value (as-is string)."""


# ---------------------------------------------------------------------------
# FieldValue envelope
# ---------------------------------------------------------------------------

@dataclass
class FieldValue:
    """Provenance wrapper for a single extracted field value.

    This is the atomic unit written to every scalar in the projection output
    when normalization is enabled.  Serialise to plain dict via .to_dict().
    """

    raw_value: Any
    normalized_value: Any = None
    normalization_rule: Optional[str] = None
    normalization_confidence: float = 0.0
    source_node_id: Optional[str] = None
    source_page: Optional[int] = None
    grounded: bool = False
    unresolved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def passthrough(cls, raw: Any, **kwargs) -> "FieldValue":
        """Identity transform: value passes through unchanged at confidence=1.0."""
        return cls(
            raw_value=raw,
            normalized_value=raw,
            normalization_rule="passthrough",
            normalization_confidence=1.0,
            unresolved=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Normalizer helpers (private, pure functions)
# ---------------------------------------------------------------------------

# Ordered by specificity — first match wins.
_AMOUNT_RULES: list[tuple[str, str, float, Any]] = [
    # (pattern, rule_name, confidence, transform_hint)
    # Full ISO currency prefix, e.g. "USD 2,230.00"
    (r"^[A-Z]{3}\s*([\d,]+\.?\d*)$", "iso_currency_prefix", 0.97, "strip_symbol_comma"),
    # Single-char symbol prefix: "$ 10" or "€10"
    (r"^[€$£¥₹]\s*([\d,\.]+)$", "symbol_prefix", 0.92, "strip_symbol_comma"),
    # Trailing symbol: "2,230.00 USD"
    (r"^([\d,]+\.?\d*)\s*[A-Z]{3}$", "iso_currency_suffix", 0.94, "strip_suffix_comma"),
    # European format: "2.230,00"
    (r"^([\d]{1,3}(?:\.\d{3})+,\d{2})$", "european_decimal", 0.88, "european"),
    # Plain comma-thousands: "2,230.00" or "2,230"
    (r"^([\d,]+\.?\d*)$", "comma_thousands", 0.82, "strip_comma"),
    # Plain integer or float
    (r"^(\d+\.?\d*)$", "plain_number", 0.99, "float_cast"),
]

_DATE_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_DMY_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$")
_DATE_MDY_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$")
_DATE_NATURAL_RE = re.compile(
    r"(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s,]+(\d{4})",
    re.IGNORECASE,
)
_DATE_NATURAL_MDY_RE = re.compile(
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{1,2})[\s,]+(\d{4})",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

CURRENCY_CODES = {"USD", "EUR", "GBP", "INR", "JPY", "CNY", "MXN", "BRL", "AUD", "CAD", "CHF", "SGD"}


def _parse_amount_str(raw: str) -> Tuple[Optional[float], str, float]:
    """Return (normalised_float, rule_name, confidence) or (None, '', 0.0)."""
    s = raw.strip()
    for pattern, rule, conf, transform in _AMOUNT_RULES:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            captured = m.group(1)
            try:
                if transform == "european":
                    # "2.230,00" → remove dots then replace comma with dot
                    val = float(captured.replace(".", "").replace(",", "."))
                elif transform in ("strip_symbol_comma", "strip_suffix_comma", "strip_comma", "comma_thousands"):
                    val = float(captured.replace(",", ""))
                else:
                    val = float(captured)
                return val, rule, conf
            except ValueError:
                continue
    return None, "", 0.0


def _parse_date_str(raw: str) -> Tuple[Optional[str], str, float]:
    """Return (iso_date_str, rule_name, confidence) or (None, '', 0.0)."""
    s = raw.strip()

    # Already ISO
    if _DATE_ISO_RE.match(s):
        return s, "iso_passthrough", 1.0

    # DMY: "06-06-2025" or "06/06/2025"  — assume DMY for ambiguous cases
    m = _DATE_DMY_RE.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}", "dmy_to_iso", 0.85

    # Natural: "06 June 2025" or "June 6, 2025"
    m = _DATE_NATURAL_RE.search(s)
    if m:
        d = int(m.group(1))
        mo = _MONTH_MAP[m.group(2)[:3].lower()]
        y = int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}", "natural_dmy_to_iso", 0.90

    m = _DATE_NATURAL_MDY_RE.search(s)
    if m:
        mo = _MONTH_MAP[m.group(1)[:3].lower()]
        d = int(m.group(2))
        y = int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}", "natural_mdy_to_iso", 0.88

    return None, "", 0.0


# ---------------------------------------------------------------------------
# Public normalizer class
# ---------------------------------------------------------------------------

class FieldNormalizer:
    """
    Confidence-gated field normalizer.

    Usage::

        normalizer = FieldNormalizer()
        fv = normalizer.normalize("USD 2,230.00", field_type_hint="amount")
        # fv.normalized_value  → 2230.0
        # fv.normalization_rule → "iso_currency_prefix"

    Ablation:
        Pass ``bypass=True`` to return a passthrough FieldValue with
        normalization_rule="passthrough" and normalized_value == raw_value.
        This lets you measure the delta between A2 (no normalization) and
        A3 (with normalization) with identical projection logic.
    """

    def __init__(self, confidence_gate: float = CONFIDENCE_GATE):
        self.confidence_gate = confidence_gate

    # ------------------------------------------------------------------
    # Typed normalizers
    # ------------------------------------------------------------------

    def normalize_amount(
        self,
        raw: Any,
        source_node_id: Optional[str] = None,
        source_page: Optional[int] = None,
    ) -> FieldValue:
        if raw is None:
            return FieldValue(raw_value=None, normalized_value=None, unresolved=True)
        raw_str = str(raw).strip()
        val, rule, conf = _parse_amount_str(raw_str)
        grounded = source_node_id is not None
        if val is not None and conf >= self.confidence_gate:
            return FieldValue(
                raw_value=raw_str,
                normalized_value=val,
                normalization_rule=rule,
                normalization_confidence=conf,
                source_node_id=source_node_id,
                source_page=source_page,
                grounded=grounded,
                unresolved=False,
            )
        # Confidence below gate → keep raw, mark unresolved
        logger.debug("normalize_amount: confidence %.2f below gate for %r", conf, raw_str)
        return FieldValue(
            raw_value=raw_str,
            normalized_value=raw_str,
            normalization_rule=rule or None,
            normalization_confidence=conf,
            source_node_id=source_node_id,
            source_page=source_page,
            grounded=grounded,
            unresolved=True,
        )

    def normalize_currency(
        self,
        raw: Any,
        source_node_id: Optional[str] = None,
        source_page: Optional[int] = None,
    ) -> FieldValue:
        if raw is None:
            return FieldValue(raw_value=None, normalized_value=None, unresolved=True)
        raw_str = str(raw).strip().upper()
        if raw_str in CURRENCY_CODES:
            return FieldValue(
                raw_value=raw_str,
                normalized_value=raw_str,
                normalization_rule="known_iso_currency",
                normalization_confidence=1.0,
                source_node_id=source_node_id,
                source_page=source_page,
                grounded=source_node_id is not None,
                unresolved=False,
            )
        # Try to extract a currency code from a mixed string like "USD 2,230"
        m = re.search(r"\b([A-Z]{3})\b", raw_str)
        if m and m.group(1) in CURRENCY_CODES:
            return FieldValue(
                raw_value=str(raw).strip(),
                normalized_value=m.group(1),
                normalization_rule="currency_extracted_from_amount",
                normalization_confidence=0.80,
                source_node_id=source_node_id,
                source_page=source_page,
                grounded=source_node_id is not None,
                unresolved=False,
            )
        # Single char symbols
        _SYMBOL_MAP = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
        for sym, code in _SYMBOL_MAP.items():
            if sym in raw_str:
                return FieldValue(
                    raw_value=str(raw).strip(),
                    normalized_value=code,
                    normalization_rule="currency_symbol_to_iso",
                    normalization_confidence=0.72,
                    source_node_id=source_node_id,
                    source_page=source_page,
                    grounded=source_node_id is not None,
                    unresolved=False,
                )
        return FieldValue(
            raw_value=str(raw).strip(),
            normalized_value=str(raw).strip(),
            normalization_rule=None,
            normalization_confidence=0.0,
            source_node_id=source_node_id,
            source_page=source_page,
            grounded=source_node_id is not None,
            unresolved=True,
        )

    def normalize_date(
        self,
        raw: Any,
        source_node_id: Optional[str] = None,
        source_page: Optional[int] = None,
    ) -> FieldValue:
        if raw is None:
            return FieldValue(raw_value=None, normalized_value=None, unresolved=True)
        raw_str = str(raw).strip()
        iso, rule, conf = _parse_date_str(raw_str)
        grounded = source_node_id is not None
        if iso and conf >= self.confidence_gate:
            return FieldValue(
                raw_value=raw_str,
                normalized_value=iso,
                normalization_rule=rule,
                normalization_confidence=conf,
                source_node_id=source_node_id,
                source_page=source_page,
                grounded=grounded,
                unresolved=False,
            )
        return FieldValue(
            raw_value=raw_str,
            normalized_value=raw_str,
            normalization_rule=rule or None,
            normalization_confidence=conf,
            source_node_id=source_node_id,
            source_page=source_page,
            grounded=grounded,
            unresolved=True,
        )

    def normalize_string(
        self,
        raw: Any,
        source_node_id: Optional[str] = None,
        source_page: Optional[int] = None,
    ) -> FieldValue:
        """Whitespace-normalised string.  Always succeeds at confidence=1.0."""
        if raw is None:
            return FieldValue(raw_value=None, normalized_value=None, unresolved=True)
        normalised = " ".join(str(raw).split())
        return FieldValue(
            raw_value=str(raw),
            normalized_value=normalised,
            normalization_rule="whitespace_normalise",
            normalization_confidence=1.0,
            source_node_id=source_node_id,
            source_page=source_page,
            grounded=source_node_id is not None,
            unresolved=False,
        )

    # ------------------------------------------------------------------
    # Dispatch entry point
    # ------------------------------------------------------------------

    def normalize(
        self,
        raw: Any,
        field_type_hint: str = "string",
        source_node_id: Optional[str] = None,
        source_page: Optional[int] = None,
        bypass: bool = False,
    ) -> FieldValue:
        """
        Dispatch normalisation by field_type_hint.

        Parameters
        ----------
        raw : Any
            The raw extracted value.
        field_type_hint : str
            One of: 'amount', 'currency', 'date', 'string'.
            Any unrecognised hint falls back to 'string'.
        source_node_id : str, optional
            Graph node identifier for provenance.
        source_page : int, optional
            1-based page number.
        bypass : bool
            If True, skip normalisation and return passthrough FieldValue.
            Set this for ablation condition A0/A1/A2 to measure the delta.
        """
        if bypass:
            return FieldValue.passthrough(
                raw,
                source_node_id=source_node_id,
                source_page=source_page,
                grounded=source_node_id is not None,
            )

        hint = field_type_hint.lower()
        if hint == "amount":
            return self.normalize_amount(raw, source_node_id, source_page)
        if hint == "currency":
            return self.normalize_currency(raw, source_node_id, source_page)
        if hint == "date":
            return self.normalize_date(raw, source_node_id, source_page)
        return self.normalize_string(raw, source_node_id, source_page)

    # ------------------------------------------------------------------
    # Bulk normalisation over a projected payload dict
    # ------------------------------------------------------------------

    # Maps field name substrings to type hints.  Evaluated in order.
    _FIELD_HINT_MAP: list[tuple[re.Pattern, str]] = [
        (re.compile(r"amount|price|cost|total|fee|rate|subtotal|tax", re.I), "amount"),
        (re.compile(r"currency|curr", re.I), "currency"),
        (re.compile(r"date|issued|expiry|validity|valid_until|due", re.I), "date"),
    ]

    def _infer_hint(self, field_name: str) -> str:
        for pattern, hint in self._FIELD_HINT_MAP:
            if pattern.search(field_name):
                return hint
        return "string"

    def normalise_payload(
        self,
        payload: dict,
        schema_json: dict,
        bypass: bool = False,
    ) -> dict:
        """
        Walk a projected extraction payload and wrap every scalar value that
        is a non-null string in a FieldValue envelope.

        Skips: list values (line_items, entities, etc.), nested dicts, and
        internal bookkeeping keys (status, error, details, reasoning_thoughts,
        page_references, confidence_score).

        The returned dict is identical in structure to the input but with
        scalar string values replaced by FieldValue.to_dict() sub-dicts.
        This is the format written to *_extraction_result.json when
        --with_normalization is active.
        """
        _SKIP_KEYS = {
            "status", "error", "details", "reasoning_thoughts",
            "page_references", "confidence_score", "schema_title",
        }
        schema_props = (schema_json or {}).get("properties", {}) if isinstance(schema_json, dict) else {}
        out = {}
        for key, value in payload.items():
            if key in _SKIP_KEYS:
                out[key] = value
                continue
            if value is None or isinstance(value, (list, dict)):
                out[key] = value
                continue
            if not isinstance(value, str):
                out[key] = value
                continue
            # Determine type hint from schema description or field name
            prop_desc = schema_props.get(key, {})
            hint = (
                prop_desc.get("x_normalize_as")  # explicit override in schema
                or self._infer_hint(key)
            )
            fv = self.normalize(value, field_type_hint=hint, bypass=bypass)
            out[key] = fv.to_dict()
        return out
