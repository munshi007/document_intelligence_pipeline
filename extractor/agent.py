"""
Extractor Agent: Librarian-Grade Agentic Reasoning
==================================================
The AI Agent that browses the HKG (Hierarchical Knowledge Graph) to perform 
Facts Extraction. Powered by the Librarian Foundation (Graph + Breadcrumbs).
"""

import logging
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, TypeVar, get_args, get_origin
from PIL import Image
from pydantic import BaseModel

from core.schemas import HierarchicalNode, Extraction, ExtractionResult
from common.vlm_client import VLMClient
from extractor.schema_definitions import LibrarianUniversalHardware, SourceEvidence
from chunker.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ExtractionFailureError(RuntimeError):
    """Raised when extraction fails with structured diagnostics."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ExtractorAgent:
    """
    State-of-the-art Agent that uses the HKG to fulfill a specific schema.
    Uses Specialists and Additive Synthesis for high-fidelity extraction.
    """
    
    SPECIALIST_HINTS = {
        "Corporate": """
Focus on financial and entity details:
1. Identify Sender (Organization/Manufacturer) and Recipient (Customer/Musterkunde).
2. Extract all 'Entities' mentioned (names, organizations).
3. Extract 'Identity' fields: Invoice No as title, Date, and Document Type (Invoice).
4. List line items or transaction-specific events in the 'Timeline'.
""",
        "Hardware": """
Focus on technical specifications:
1. Identify product name, article number (Art. No), and manufacturer.
2. Extract ALL 'parameters' from parameter tables — every row is one TechParameter entry.
   Include electrical (supply voltage, current, power), mechanical (dimensions, weight),
   and environmental (temperature range, protection class IP xx) parameters.
3. Detail ALL connectors: for EACH connector extract EVERY pin assignment (pin number, signal, function).
4. Capture ALL LED diagnostic states and their meanings.
5. Note any standards or certifications mentioned.
""",
        # Industrial docs are routed here — same hints as Hardware
        "Industrial": """
Focus on technical specifications:
1. Identify product name, article number (Art. No), and manufacturer.
2. Extract ALL 'parameters' from parameter tables — every row is one TechParameter entry.
   Include electrical (supply voltage, current, power), mechanical (dimensions, weight),
   and environmental (temperature range, protection class IP xx) parameters.
3. Detail ALL connectors: for EACH connector extract EVERY pin assignment (pin number, signal, function).
4. Capture ALL LED diagnostic states and their meanings.
5. Note any standards or certifications mentioned.
""",
        "Industrial_datasheet": """
You are extracting from a TECHNICAL DATASHEET. Be exhaustive and precise:
1. Identity: extract product name, article number (Art. No / Art.-Nr.), and manufacturer.
2. Parameters: extract EVERY row of every parameter table as a TechParameter.
   - Electrical: supply voltage (e.g. "18...30 V DC"), current consumption, short-circuit protection
   - Mechanical: housing dimensions (L×W×H in mm), weight (g or kg), housing material
   - Environmental: operating temperature (e.g. "-25...+70 °C"), protection class (e.g. "IP67")
3. Connectors: for EVERY connector (M12, M8, RJ45, etc.) list ALL pins with signal and function.
4. Diagnostics: for EVERY LED list EVERY state (color + behavior) and its diagnostic meaning.
5. Standards: list all certifications and standards (CE, UL, IEC numbers, PROFINET, IO-Link, etc.).
""",
        "Industrial_manual": """
Focus on technical specifications within the manual:
1. Identify product name, article number, and manufacturer.
2. Extract ALL parameters from technical data sections/tables.
3. Detail connector pinouts and wiring diagrams.
4. Capture LED, error code, and diagnostic information.
5. Note installation steps and configuration parameters.
""",
        "Invoice": """
Focus on invoice fields with line items:
1. Header: invoice number, date, due date if present.
2. Parties: buyer (BILL TO / SHIP TO / TO) and seller (FROM / SOLD BY) — extract name + full address.
3. Items: EVERY line — description, quantity, unit price, line total. Preserve exact strings for descriptions.
4. Totals: subtotal, tax/VAT lines, shipping, grand total.
5. Currency: infer from $/€/£/¥ symbols in totals; populate currency field if present in schema.
""",
        "Logistics": """
Focus on freight quote / bill-of-lading / shipping tariff fields:
1. Identity: quote number / reference, issuing carrier, recipient/customer, quote date, validity dates.
2. Route: origin and destination (cities/countries), ports of loading and discharge, haulage type.
3. Commodity: cargo description, container types/sizes (e.g. 20'STD, 40'HC), weight/volume if given.
4. Charges: EVERY line in every rate table — description (Seafreight, Bunker, Surcharge, Document Fee…), value, currency, and the unit (per container / per BoL / lumpsum) when stated.
5. Currency: infer from $/€/£/¥ symbols or three-letter codes (USD, EUR, GBP, ARS…); populate currency on each charge.
""",
        "General": """
Focus on comprehensive narrative capturing:
1. Extract all names and organizations in 'Entities'.
2. Create a detailed 'Timeline' of any dates/events mentioned.
3. Provide a high-level narrative 'Summary' and 'Purpose'.
"""
    }

    # Domain label canonicalisation. The Discovery agent may return either
    # capitalised ("Logistics") or lowercased ("logistics") labels depending on
    # whether the LLM path or the heuristic path produced the result, and
    # aliases (Industrial_datasheet/_manual/_product_pdf) all share one hint.
    _DOMAIN_ALIAS = {
        "industrial_datasheet": "Industrial_datasheet",
        "industrial_manual":    "Industrial_manual",
        "industrial_product_pdf": "Industrial",
        "industrial":           "Industrial",
        "hardware":             "Hardware",
        "logistics":            "Logistics",
        "invoice":              "Invoice",
        "corporate":            "Corporate",
        "medical_record":       "General",
        "academic":             "General",
        "legal":                "General",
        "general":              "General",
    }

    def _specialist_hint(self, domain: str) -> str:
        """Case- and alias-tolerant lookup into SPECIALIST_HINTS."""
        if not domain:
            return f"Follow the JSON schema exactly to extract all entities, properties, and lists relevant to the document."
        if domain in self.SPECIALIST_HINTS:
            return self.SPECIALIST_HINTS[domain]
        canon = self._DOMAIN_ALIAS.get(domain.lower())
        if canon and canon in self.SPECIALIST_HINTS:
            return self.SPECIALIST_HINTS[canon]
        return f"Follow the JSON schema exactly to extract all entities, properties, and lists relevant to the {domain} domain."

    def __init__(self, model_id: str = "gpt-4o", observer: Optional[Any] = None):
        self.model_id = model_id
        self.client = VLMClient(config={"model": model_id})
        self.client.observer = observer
        logger.info(f"ExtractorAgent initialized with Librarian Teacher: {model_id}")

    def _build_schema_aligned_hint(self, response_model: Type[T], domain: str) -> str:
        """Auto-generate extraction instructions FROM the actual schema fields."""
        def describe_recursive(model_type, indent=0):
            lines = []
            prefix = "  " * indent
            
            # Handle list/optional wrappers to get the core model
            import typing
            origin = typing.get_origin(model_type)
            if origin in (list, typing.Union, Optional):
                args = typing.get_args(model_type)
                for arg in args:
                    if hasattr(arg, "model_fields"):
                        lines.extend(describe_recursive(arg, indent))
                return lines

            if not hasattr(model_type, "model_fields"):
                return lines

            for f_name, f_info in model_type.model_fields.items():
                if f_name in ('reasoning_thoughts', 'page_references', 'confidence_score'):
                    continue
                f_desc = f_info.description or f_name.replace('_', ' ')
                f_type = str(f_info.annotation).replace("typing.", "").replace("NoneType", "").strip("| ")
                
                lines.append(f"{prefix}- **{f_name}** ({f_type}): {f_desc}")
                
                # Recurse if it's a sub-model
                ann = f_info.annotation
                sub_origin = typing.get_origin(ann)
                if sub_origin in (list, typing.Union, Optional):
                    sub_args = typing.get_args(ann)
                    for sub_arg in sub_args:
                        if hasattr(sub_arg, "model_fields"):
                            lines.extend(describe_recursive(sub_arg, indent + 1))
                elif hasattr(ann, "model_fields"):
                    lines.extend(describe_recursive(ann, indent + 1))
            return lines

        field_descriptions = describe_recursive(response_model)

        field_list = "\n".join(field_descriptions)
        return f"""You are the Visual Librarian and Technical Specialist.
Extract ONLY the following fields (these are the ONLY fields in the schema):
{field_list}

### EXTRACTION RULES (STRICT):
1. **SOURCE FIDELITY (MOST IMPORTANT)**: Extract ONLY values that actually appear in the SOURCE CONTENT below. NEVER infer, complete, or add values from your own knowledge of what a typical {domain} document usually contains. If the source does not state a value, leave that field empty/null and omit the list item entirely — do NOT invent a plausible one. An empty-but-true result is correct; a complete-but-invented one is a failure.
2. **SCHEMA IS LAW**: DO NOT add fields from your own knowledge of what this kind of document usually contains. USE ONLY the exact field names listed above.
3. **STRUCTURAL INTEGRITY**: Your response MUST follow the field structure exactly as defined above at the root level.
4. **NO HALLUCINATED WRAPPERS**: DO NOT wrap these fields in sub-objects like 'identity', 'metadata', or 'data' unless specified in the schema.
5. **CAPTURE WHAT IS PRESENT**: Capture EVERY relevant row and data point that IS in the source — especially technical tables (like pin assignments). But never manufacture rows or values that are absent from the source (see Rule 1): completeness means missing nothing real, not filling everything in.
6. **CURRENCY INFERENCE**: When a `value` (or `amount` / `unit_price` / `total`) field contains a currency symbol, ALSO populate the sibling `currency` field: "$"→"USD", "€"→"EUR", "£"→"GBP", "¥"→"JPY". Keep the symbol in the value.
7. **REASONING**: Briefly explain your extraction logic in the `reasoning_thoughts` field.

### METADATA FIELDS (always populate when present in schema):
- `page_references`: list of page numbers (integers) where the extracted facts appeared. The SOURCE CONTENT contains `<!-- page:N -->` markers — read them and include each page you drew evidence from.
- `source_evidence`: list of `{{text_snippet, page_number, confidence}}` items quoting the exact source text supporting your top extractions. Aim for 3–6 items, each ≤ 200 chars. Skip if the field is not in the schema."""

    @staticmethod
    def _append_jsonl(path: Optional[str], payload: Dict[str, Any]) -> None:
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _score_quality(result: BaseModel, response_model: Type[BaseModel]) -> Dict[str, Any]:
        result_dump = result.model_dump()
        populated_fields_count = 0
        for value in result_dump.values():
            if value is None:
                continue
            if isinstance(value, (list, dict)) and len(value) == 0:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            populated_fields_count += 1

        required = response_model.model_json_schema().get("required", [])
        required_fields_missing = []
        for field in required:
            value = result_dump.get(field)
            if value is None:
                required_fields_missing.append(field)
            elif isinstance(value, (list, dict)) and len(value) == 0:
                required_fields_missing.append(field)
            elif isinstance(value, str) and value.strip() == "":
                required_fields_missing.append(field)

        return {
            "populated_fields_count": populated_fields_count,
            "required_fields_missing": required_fields_missing,
        }

    @staticmethod
    def _is_non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    @staticmethod
    def _default_for_schema_type(prop_schema: Dict[str, Any]) -> Any:
        ptype = prop_schema.get("type")
        if isinstance(ptype, list):
            if "null" in ptype:
                return None
            if "array" in ptype:
                return []
            if "object" in ptype:
                return {}
            return None
        if ptype == "array":
            return []
        if ptype == "object":
            return {}
        if ptype in ("string", "number", "integer", "boolean"):
            return None
        return None

    @staticmethod
    def _extract_line_items_from_tables(tables_markdown: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for table in tables_markdown or []:
            for line in table.splitlines():
                row = line.strip()
                if not row.startswith("|"):
                    continue
                if set(row.replace("|", "").replace("-", "").strip()) == set():
                    continue
                cols = [c.strip() for c in row.split("|")[1:-1]]
                if len(cols) < 2:
                    continue
                if cols[0].lower().startswith("per "):
                    continue
                if cols[0].lower() in {"--", "curr.", "tk20", "tk220"}:
                    continue
                description = cols[0] if cols[0] else None
                amount = cols[-1] if cols[-1] else None
                if description or amount:
                    items.append({"description": description, "amount": amount})
        return items

    @staticmethod
    def _verify_string_spans(
        record: Any,
        source_markdown: str,
        *,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Verify that string-valued fields in `record` appear in `source_markdown`,
        and snap them to the closest source span when they don't (e.g. character-
        level drops produced by the extractor LLM such as 'TIBAANAMA' for
        'TIBA PANAMA').

        Numeric leaves are verified too (locale-aware: '1.000,00' and
        '1,000.00' both parse to 1000.0): a number must match a numeric token
        in the source — preferentially in the same source LINE as a grounded
        string sibling (its table row). A flagged number with a unique close
        digit-match in its row is snapped to the row's value
        (kind='numeric_snap'); otherwise it is flagged with kind='numeric'.

        Skips meta/enum/reasoning fields. Returns audit stats:
          {checked, verified, repaired: [{path, before, after, ratio}],
           flagged:  [{path, value, best_ratio}],
           provenance: [{path, snippet, page, ratio}]}
        `provenance` records the page (from <!-- page:N --> markers in the
        source markdown) for every verified/repaired field — used by the
        caller to auto-fill `page_references` and `source_evidence`.

        Mutates `record` in place.
        """
        import re
        import difflib

        SKIP_FIELDS = {
            "reasoning_thoughts", "confidence_score", "page_references",
            "domain", "currency", "param_type", "is_high_density",
        }
        PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")

        def normalize(s: Any) -> str:
            if not isinstance(s, str):
                return ""
            t = re.sub(r"<!--[^>]*-->", " ", s)
            t = re.sub(r"\s+", " ", t)
            return t.lower().strip()

        def find_page_for_span(span: str) -> Optional[int]:
            """Locate `span` in the ORIGINAL source (case-insensitive) and
            return the page number from the closest `<!-- page:N -->` marker
            preceding it."""
            if not span or not source_markdown:
                return None
            try:
                m = re.search(re.escape(span), source_markdown, re.IGNORECASE)
            except re.error:
                return None
            if not m:
                return None
            markers = list(PAGE_MARKER.finditer(source_markdown[: m.start() + 1]))
            return int(markers[-1].group(1)) if markers else None

        norm_source = normalize(source_markdown or "")

        def recover_source_case(value: str) -> Optional[str]:
            """Locate `value` in the ORIGINAL source (case-insensitively,
            tolerating whitespace runs and stripped <!-- --> comments between
            tokens) and return the source's own rendering. The source is
            ground truth for casing: a value that matches it only
            case-insensitively was case-mangled by the model."""
            tokens = normalize(value).split()
            if not tokens:
                return None
            sep = r"(?:\s|<!--[^>]*-->)+"
            pattern = sep.join(re.escape(t) for t in tokens)
            try:
                m = re.search(pattern, source_markdown or "", re.IGNORECASE)
            except re.error:
                return None
            if not m:
                return None
            return re.sub(r"<!--[^>]*-->", " ", m.group(0)).strip()

        def fuzzy_find(needle: str) -> tuple:
            """Return (best_ratio, repaired_str_or_None) for needle vs source."""
            n = normalize(needle)
            if len(n) < 3 or not norm_source:
                return 0.0, None
            if n in norm_source:
                # Verbatim under normalization; caller (walk) handles
                # case restoration for this branch before reaching here.
                return 1.0, None
            L = len(n)
            best_ratio = 0.0
            best_span = None
            # Search windows of length L±3 over the source
            for wl in range(max(3, L - 3), L + 4):
                if wl > len(norm_source):
                    continue
                for i in range(0, len(norm_source) - wl + 1):
                    cand = norm_source[i:i + wl]
                    # Cheap pre-filter: at least one common 3-gram
                    if not any(n[k:k+3] in cand for k in range(0, len(n) - 2, 2)):
                        continue
                    ratio = difflib.SequenceMatcher(None, n, cand).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (i, wl)
                        if ratio == 1.0:
                            break
                if best_ratio == 1.0:
                    break
            if best_span is None:
                return best_ratio, None
            i, wl = best_span
            # Recover the original-case span from source_markdown via case-insensitive search
            cand_lower = norm_source[i:i + wl]
            m = re.search(re.escape(cand_lower), source_markdown or "", re.IGNORECASE)
            return best_ratio, (m.group(0).strip() if m else cand_lower)

        stats: Dict[str, Any] = {
            "checked": 0, "verified": 0,
            "repaired": [], "flagged": [], "provenance": [],
        }

        # ── numeric verification helpers ──
        NUM_TOKEN = re.compile(r"(?<![\w.,])[-+]?\d(?:[\d.,]*\d)?(?!\w)")

        def parse_token(tok: str) -> set:
            """All plausible parses of a source number token. Locale-agnostic:
            '1.000,00' → {1000.0}; '1,80' → {1.8, 180.0} (decimal comma vs
            thousands comma) — ambiguity is resolved by value equality."""
            t = tok.strip().replace(" ", "")
            outs: set = set()

            def _try(s: str) -> None:
                try:
                    outs.add(float(s))
                except ValueError:
                    pass

            if "," in t and "." in t:
                if t.rfind(",") > t.rfind("."):
                    _try(t.replace(".", "").replace(",", "."))   # 1.000,00
                else:
                    _try(t.replace(",", ""))                     # 1,000.00
            elif "," in t:
                _try(t.replace(",", "."))                        # decimal comma
                _try(t.replace(",", ""))                         # thousands comma
            elif "." in t:
                _try(t)                                          # decimal point
                _try(t.replace(".", ""))                         # thousands point
            else:
                _try(t)
            return outs

        def numbers_in(text: str) -> List[float]:
            vals: List[float] = []
            for m in NUM_TOKEN.finditer(text or ""):
                vals.extend(parse_token(m.group(0)))
            return vals

        def num_eq(a: float, b: float) -> bool:
            return abs(a - b) <= max(1e-6, abs(b) * 1e-6)

        def anchor_line(parent: Dict[str, Any]) -> Optional[str]:
            """The source LINE containing a string sibling of a numeric leaf —
            i.e. the table row this number belongs to."""
            for sv in parent.values():
                if not (isinstance(sv, str) and len(sv.strip()) >= 3):
                    continue
                probe = sv.strip().splitlines()[0][:80]
                if len(probe) < 3:
                    continue
                m = re.search(re.escape(probe), source_markdown or "", re.IGNORECASE)
                if not m:
                    continue
                s = (source_markdown or "").rfind("\n", 0, m.start()) + 1
                e = (source_markdown or "").find("\n", m.end())
                return (source_markdown or "")[s: e if e != -1 else None]
            return None

        doc_nums: Optional[List[float]] = None

        def check_number(parent: Dict[str, Any], key: str, val: Any, full_path: str) -> None:
            nonlocal doc_nums
            stats["checked"] += 1
            target = float(val)
            row = anchor_line(parent)
            row_nums = numbers_in(row) if row else []
            if any(num_eq(target, c) for c in row_nums):
                stats["verified"] += 1
                return
            if doc_nums is None:
                doc_nums = numbers_in(source_markdown or "")
            if any(num_eq(target, c) for c in doc_nums):
                stats["verified"] += 1
                return
            # Row-scoped digit snap: the model produced a number close to —
            # but not equal to — one in its own row (digit insertion/drop,
            # e.g. 2243 for 243). Only safe inside the row: doc-wide there
            # are far too many candidate numbers.
            scored: List[tuple] = []
            if row_nums:
                td = re.sub(r"\D", "", f"{target:g}")
                for c in set(row_nums):
                    cd = re.sub(r"\D", "", f"{c:g}")
                    if not cd or not td:
                        continue
                    scored.append(
                        (difflib.SequenceMatcher(None, td, cd).ratio(), c)
                    )
                scored.sort(reverse=True)
                if scored and scored[0][0] >= 0.6 and (
                    len(scored) == 1 or scored[0][0] > scored[1][0]
                ):
                    best_ratio, best_val = scored[0]
                    repaired_val: Any = (
                        int(best_val)
                        if isinstance(val, int) and float(best_val).is_integer()
                        else best_val
                    )
                    parent[key] = repaired_val
                    stats["verified"] += 1
                    stats["repaired"].append({
                        "path": full_path, "before": val, "after": repaired_val,
                        "ratio": round(best_ratio, 3), "kind": "numeric_snap",
                    })
                    return
            stats["flagged"].append({
                "path": full_path, "value": val,
                "best_ratio": round(scored[0][0], 3) if scored else 0.0,
                "kind": "numeric",
            })

        def is_candidate(key: str, val: Any) -> bool:
            if key in SKIP_FIELDS or not isinstance(val, str):
                return False
            return len(val.strip()) >= 3

        def is_numeric_candidate(key: str, val: Any) -> bool:
            return (
                key not in SKIP_FIELDS
                and isinstance(val, (int, float))
                and not isinstance(val, bool)
            )

        def record_provenance(path: str, span: str, ratio: float) -> None:
            page = find_page_for_span(span)
            if page is not None:
                stats["provenance"].append({
                    "path": path,
                    "snippet": span[:200],
                    "page": page,
                    "ratio": round(ratio, 3),
                })

        def walk(node: Any, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in list(node.items()):
                    full = f"{path}.{k}" if path else k
                    if is_candidate(k, v):
                        stats["checked"] += 1
                        if normalize(v) in norm_source:
                            stats["verified"] += 1
                            restored = recover_source_case(v)
                            if restored and restored != v:
                                # Same text per the source, but the model
                                # mangled case/whitespace — snap to source.
                                node[k] = restored
                                # kind=case_restore: already counted in
                                # `verified` (verbatim modulo casing) — summary
                                # consumers must not add it to grounded again.
                                stats["repaired"].append({
                                    "path": full, "before": v,
                                    "after": restored, "ratio": 1.0,
                                    "kind": "case_restore",
                                })
                                record_provenance(full, restored, 1.0)
                            else:
                                record_provenance(full, v, 1.0)
                        else:
                            ratio, repaired = fuzzy_find(v)
                            if (repaired
                                and ratio >= threshold
                                and normalize(repaired) != normalize(v)):
                                node[k] = repaired
                                # The snapped value IS grounded — count it in
                                # `verified` so that number is the single
                                # complete grounded count. `repaired` is the
                                # audit trail, never added to pass rates
                                # (post-retry refreshes re-verify snapped
                                # values verbatim; summing the two double-
                                # counted and pushed pass_rate above 1.0).
                                stats["verified"] += 1
                                stats["repaired"].append({
                                    "path": full, "before": v,
                                    "after": repaired, "ratio": round(ratio, 3),
                                    "kind": "fuzzy_snap",
                                })
                                record_provenance(full, repaired, ratio)
                            else:
                                stats["flagged"].append({
                                    "path": full, "value": v,
                                    "best_ratio": round(ratio, 3),
                                })
                    elif is_numeric_candidate(k, v):
                        check_number(node, k, v, full)
                    else:
                        walk(v, full)
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    walk(item, f"{path}[{idx}]")

        walk(record)
        return stats

    # ── Recall coverage (omission detection + targeted list recovery) ──
    #
    # Every check above is a PRECISION check: it judges values that are
    # present. None of them can see an omission — a model that returns
    # items:[] with a plausible rationalization scores populated=1.0 and
    # grounding pass_rate 1.0. The coverage pass closes that blind spot by
    # asking the inverse question: which source-table rows never made it
    # into the record at all?

    @staticmethod
    def _markdown_tables(source_markdown: str) -> List[Dict[str, Any]]:
        """Parse pipe-tables out of the source markdown.

        Returns [{'header': [cells], 'rows': [[cells]], 'text': block}].
        The first non-separator row is the header; `---` rows are dropped.
        """
        blocks: List[List[str]] = []
        cur: List[str] = []
        for line in (source_markdown or "").splitlines():
            if line.lstrip().startswith("|"):
                cur.append(line)
            else:
                if len(cur) >= 2:
                    blocks.append(cur)
                cur = []
        if len(cur) >= 2:
            blocks.append(cur)

        sep = re.compile(r":?-{2,}:?$")
        tables: List[Dict[str, Any]] = []
        for block in blocks:
            header: Optional[List[str]] = None
            rows: List[List[str]] = []
            for ln in block:
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if cells and all(sep.fullmatch(c) for c in cells if c):
                    continue
                if header is None:
                    header = cells
                else:
                    rows.append(cells)
            if header is not None:
                tables.append({"header": header, "rows": rows, "text": "\n".join(block)})
        return tables

    @staticmethod
    def _norm_text(s: Any) -> str:
        return re.sub(r"\s+", " ", str(s or "")).lower().strip()

    def _undercovered_tables(
        self,
        record: Dict[str, Any],
        source_markdown: str,
        *,
        min_rows: int = 3,
        threshold: float = 0.5,
    ) -> tuple:
        """Return (flags, tables): source tables whose data rows mostly do not
        appear anywhere in the extracted record. A row counts as covered when
        any of its cells (≥3 chars, normalized) occurs in the record text."""
        tables = self._markdown_tables(source_markdown)
        blob = self._norm_text(json.dumps(record, ensure_ascii=False, default=str))
        flags: List[Dict[str, Any]] = []
        for ti, t in enumerate(tables):
            rows = [r for r in t["rows"] if any(c for c in r)]
            if len(rows) < min_rows:
                continue
            covered = sum(
                1 for r in rows
                if any(len(self._norm_text(c)) >= 3 and self._norm_text(c) in blob for c in r)
            )
            if covered / len(rows) < threshold:
                flags.append({
                    "table_index": ti,
                    "data_rows": len(rows),
                    "covered_rows": covered,
                    "header": (t["header"] or [])[:8],
                })
        return flags, tables

    @staticmethod
    def _header_field_affinity(header: List[str], item_model: Type[BaseModel]) -> float:
        """Fraction of the item model's field names whose tokens appear in the
        table header. Derived entirely from the discovered schema and the
        document's own header text — no fixed vocabulary."""
        header_tokens: set = set()
        for cell in header or []:
            header_tokens |= set(re.findall(r"[a-zA-Z]{3,}", str(cell).lower()))
        fields = list(getattr(item_model, "model_fields", {}).keys())
        if not header_tokens or not fields:
            return 0.0
        hits = sum(
            1 for f in fields
            if any(tok in header_tokens for tok in re.findall(r"[a-z]{3,}", f.lower()))
        )
        return hits / len(fields)

    @staticmethod
    def _list_object_fields(response_model: Type[BaseModel]) -> Dict[str, Type[BaseModel]]:
        """Top-level schema fields typed List[<pydantic model>] → item model."""
        import typing
        out: Dict[str, Type[BaseModel]] = {}
        for f_name, f_info in response_model.model_fields.items():
            ann = f_info.annotation
            origin = typing.get_origin(ann)
            if origin is typing.Union:
                for a in typing.get_args(ann):
                    if typing.get_origin(a) is list:
                        ann, origin = a, list
                        break
            if origin is list:
                args = typing.get_args(ann)
                if args and hasattr(args[0], "model_fields"):
                    out[f_name] = args[0]
        return out

    def _recover_list_coverage(
        self,
        record: Dict[str, Any],
        response_model: Type[BaseModel],
        flags: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        trace_context: Optional[Dict[str, Any]] = None,
        max_recoveries: int = 2,
    ) -> List[Dict[str, Any]]:
        """Targeted re-extraction of EMPTY list fields from under-covered
        tables. One focused LLM call per (field, table); the table is matched
        to the field by header↔field-name affinity, and every recovered item
        must ground in the table text or it is dropped. Non-empty lists are
        never mutated — they are only reported via the coverage flags."""
        from pydantic import create_model

        audit: List[Dict[str, Any]] = []
        empty_fields = {
            f: m for f, m in self._list_object_fields(response_model).items()
            if not record.get(f)
        }
        if not empty_fields or not flags:
            return audit

        used_tables: set = set()
        for f_name, item_model in list(empty_fields.items())[:max_recoveries]:
            scored = sorted(
                (
                    (self._header_field_affinity(tables[fl["table_index"]]["header"], item_model),
                     fl["data_rows"], fl["table_index"])
                    for fl in flags if fl["table_index"] not in used_tables
                ),
                reverse=True,
            )
            scored = [s for s in scored if s[0] > 0]
            if not scored:
                continue
            affinity, n_rows, ti = scored[0]
            table = tables[ti]
            used_tables.add(ti)

            wrapper = create_model("RecoveredList", items=(List[item_model], []))
            field_lines = "\n".join(
                f"- {fn}: {fi.description or fn.replace('_', ' ')}"
                for fn, fi in item_model.model_fields.items()
            )
            prompt = f"""Extract EVERY data row of the table below as one object.
Fields per object:
{field_lines}

RULES:
1. One object per table data row — do not skip, merge, or abbreviate rows.
2. Use ONLY values visible in the table. Never invent values.
3. Output a JSON object with a single key "items" holding the array.

TABLE:
{table['text']}
"""
            result = self.client.generate_structured(
                image=None,
                prompt=prompt,
                response_model=wrapper,
                max_tokens=4096,
                trace_dir=(trace_context or {}).get("trace_dir"),
                trace_key=f"coverage_recovery_{f_name}",
            )
            items = list(getattr(result, "items", []) or []) if result else []
            table_norm = self._norm_text(table["text"])
            accepted: List[Dict[str, Any]] = []
            for it in items:
                dump = it.model_dump()
                str_vals = [v for v in dump.values() if isinstance(v, str) and len(v.strip()) >= 3]
                if str_vals and not any(self._norm_text(v) in table_norm for v in str_vals):
                    continue  # none of the item's strings exist in the table — fabricated
                accepted.append(dump)
            accepted = accepted[:n_rows]
            if accepted:
                record[f_name] = accepted
                logger.info(
                    f"      [Coverage] Recovered {len(accepted)} item(s) into "
                    f"'{f_name}' from an under-covered table ({n_rows} rows, "
                    f"affinity {affinity:.2f})."
                )
            audit.append({
                "field": f_name, "table_index": ti, "table_rows": n_rows,
                "model_items": len(items), "accepted": len(accepted),
                "affinity": round(affinity, 3),
            })
        return audit

    def _audit_list_coverage(
        self,
        record: Dict[str, Any],
        response_model: Type[BaseModel],
        source_markdown: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Detect under-covered source tables, recover empty list fields from
        them, then re-audit. Returns the coverage block for grounding stats
        (None when the source has no tables worth checking)."""
        if not source_markdown:
            return None
        flags, tables = self._undercovered_tables(record, source_markdown)
        tables_checked = sum(
            1 for t in tables if len([r for r in t["rows"] if any(r)]) >= 3
        )
        if not tables_checked:
            return None
        if not flags:
            return {"tables_checked": tables_checked, "undercovered": [], "recovered": []}
        logger.warning(
            f"      [Coverage] {len(flags)} source table(s) under-covered by the extraction."
        )
        recovered = self._recover_list_coverage(
            record, response_model, flags, tables, trace_context
        )
        if any(r.get("accepted") for r in recovered):
            flags, _ = self._undercovered_tables(record, source_markdown)
        return {
            "tables_checked": tables_checked,
            "undercovered": flags,
            "recovered": recovered,
        }

    def project_to_schema(
        self,
        source_payload: Dict[str, Any],
        target_schema_json: Dict[str, Any],
        context_markdown: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reshape model output to the runtime schema contract — structural only.

        This pass copies what the extractor already produced into the exact
        field names the target schema declares, applying ONLY domain-agnostic
        structural reshaping:

          - direct passthrough when the model already emitted the field
          - well-known synonym passthrough (summary ↔ reasoning_thoughts)
          - structural carry-through / transforms over already-extracted
            material (document_title from the model's `identity.title`;
            line_items derived from the model's extracted tables; entities
            and tables_markdown carried through)

        It NEVER guesses a field's value from the raw document via field-name
        keyed regexes or string matches. That responsibility belongs to the
        source-grounded retry pass (`_retry_flagged_and_null_strings`), which
        is schema-agnostic and verifies every filled value against the source
        before accepting it. Any field left empty here flows to that pass
        downstream — so removing the old per-field heuristic ladder (which
        baked in document-specific vendor names and domain-biased number
        regexes) costs us nothing but the fabrication risk it carried.
        """
        properties = (target_schema_json or {}).get("properties", {})
        if not isinstance(properties, dict) or not properties:
            return source_payload

        identity = source_payload.get("identity") if isinstance(source_payload.get("identity"), dict) else {}
        entities = source_payload.get("entities") if isinstance(source_payload.get("entities"), list) else []
        tables_markdown = source_payload.get("tables_markdown") if isinstance(source_payload.get("tables_markdown"), list) else []

        out: Dict[str, Any] = {}
        for field_name, field_schema in properties.items():
            direct = source_payload.get(field_name)
            if self._is_non_empty(direct):
                out[field_name] = direct
                continue

            # Structural-only reshaping of material the model already produced.
            # No raw-document content guessing lives here by design.
            inferred = None
            if field_name == "document_title":
                inferred = identity.get("title")
            elif field_name == "summary":
                inferred = source_payload.get("summary") or source_payload.get("reasoning_thoughts")
            elif field_name == "line_items":
                inferred = self._extract_line_items_from_tables(tables_markdown)
            elif field_name == "entities":
                inferred = entities
            elif field_name == "tables_markdown":
                inferred = tables_markdown

            if self._is_non_empty(inferred):
                out[field_name] = inferred
            else:
                out[field_name] = self._default_for_schema_type(field_schema if isinstance(field_schema, dict) else {})

        return out

    @staticmethod
    def _infer_currency_inplace(record: Any) -> int:
        """
        Walk a nested dict/list and fill blank `currency` siblings next to
        a `value` / `amount` / `unit_price` / `total` field that contains
        a currency symbol. Returns the number of fields filled.

        Pure-Python, idempotent, and only writes when the sibling exists
        in the schema (does not invent fields).
        """
        symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
        money_fields = ("value", "amount", "unit_price", "total", "price")
        count = 0

        def walk(node: Any) -> None:
            nonlocal count
            if isinstance(node, dict):
                if "currency" in node and not ExtractorAgent._is_non_empty(node.get("currency")):
                    for mf in money_fields:
                        v = node.get(mf)
                        if v is None:
                            continue
                        s = str(v)
                        for sym, code in symbols.items():
                            if sym in s:
                                node["currency"] = code
                                count += 1
                                break
                        if node.get("currency"):
                            break
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(record)
        return count

    # ------------------------------------------------------------------
    # Targeted-retry helpers (post-hoc span extraction)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_path_segments(path: str) -> List[Any]:
        """
        Split a verifier-produced path like 'invoice.items[0].description'
        into ['invoice', 'items', 0, 'description'].
        """
        import re
        segments: List[Any] = []
        if not path:
            return segments
        for part in path.split("."):
            m = re.match(r"^([^\[\]]*)((?:\[\d+\])*)$", part)
            if not m:
                segments.append(part)
                continue
            name, idx_chunk = m.group(1), m.group(2)
            if name:
                segments.append(name)
            for idx in re.findall(r"\[(\d+)\]", idx_chunk):
                segments.append(int(idx))
        return segments

    @staticmethod
    def _get_at_path(obj: Any, path: str) -> Any:
        """Resolve a dotted/indexed path against a nested dict/list. None if any segment misses."""
        cur = obj
        for seg in ExtractorAgent._parse_path_segments(path):
            if cur is None:
                return None
            if isinstance(seg, int):
                if isinstance(cur, list) and 0 <= seg < len(cur):
                    cur = cur[seg]
                else:
                    return None
            else:
                if isinstance(cur, dict):
                    cur = cur.get(seg)
                else:
                    return None
        return cur

    @staticmethod
    def _set_at_path(obj: Any, path: str, value: Any) -> bool:
        """
        Set a value at a dotted/indexed path. Creates intermediate dicts
        if the parent path contains nulls (so wholly-null sub-objects can
        be populated). List indices must already exist — we don't grow lists.
        """
        segs = ExtractorAgent._parse_path_segments(path)
        if not segs:
            return False
        cur = obj
        for i, seg in enumerate(segs[:-1]):
            next_seg = segs[i + 1]
            if isinstance(seg, int):
                if not isinstance(cur, list) or not (0 <= seg < len(cur)):
                    return False
                cur = cur[seg]
            else:
                if not isinstance(cur, dict):
                    return False
                if seg not in cur or cur[seg] is None:
                    cur[seg] = [] if isinstance(next_seg, int) else {}
                cur = cur[seg]
        last = segs[-1]
        if isinstance(last, int):
            if isinstance(cur, list) and 0 <= last < len(cur):
                cur[last] = value
                return True
            return False
        if isinstance(cur, dict):
            cur[last] = value
            return True
        return False

    @staticmethod
    def _unwrap_model_class(annotation: Any) -> Optional[Type[BaseModel]]:
        """Drill through Optional[X] / Union[X, None] / List[X] to find a BaseModel class, if any."""
        import typing
        if annotation is None:
            return None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin in (typing.Union, list):
            for a in args:
                if a is type(None):
                    continue
                cls = ExtractorAgent._unwrap_model_class(a)
                if cls is not None:
                    return cls
        return None

    @staticmethod
    def _resolve_field_info(root_model: Type[BaseModel], path: str) -> Optional[Any]:
        """
        Walk Pydantic .model_fields by the path segments and return the FieldInfo
        for the leaf, or None if any segment can't be resolved.
        """
        segs = ExtractorAgent._parse_path_segments(path)
        cur_model = root_model
        cur_field_info = None
        for seg in segs:
            if isinstance(seg, int):
                # List index — current annotation is List[X]; drill in
                if cur_field_info is None:
                    return None
                cur_model = ExtractorAgent._unwrap_model_class(cur_field_info.annotation)
                continue
            if cur_model is None or not hasattr(cur_model, "model_fields"):
                return None
            cur_field_info = cur_model.model_fields.get(seg)
            if cur_field_info is None:
                return None
            # Pre-drill: if next segment is also a string, move into sub-model
            next_model = ExtractorAgent._unwrap_model_class(cur_field_info.annotation)
            if next_model is not None:
                cur_model = next_model
        return cur_field_info

    @staticmethod
    def _description_for_path(root_model: Type[BaseModel], path: str) -> Optional[str]:
        fi = ExtractorAgent._resolve_field_info(root_model, path)
        return getattr(fi, "description", None) if fi is not None else None

    @staticmethod
    def _type_hint_for_path(root_model: Type[BaseModel], path: str) -> Optional[str]:
        """Return a short human-readable type tag for the leaf at `path`."""
        import typing
        fi = ExtractorAgent._resolve_field_info(root_model, path)
        if fi is None:
            return None
        ann = fi.annotation
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)
        # Unwrap Optional / Union[X, None]
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                ann = non_none[0]
        if ann is str:
            return "string"
        if ann is int:
            return "integer"
        if ann is float:
            return "number"
        if ann is bool:
            return "boolean"
        return None

    @staticmethod
    def _anchored_source(
        source: str,
        path: str,
        current_value: Any,
        window: int = 400,
        max_chars: int = 12000,
    ) -> str:
        """
        Return a focused slice of `source` around the most likely position of the
        target value. Strategies, in order:
          1. If `current_value` is a non-empty string, fuzzy-find its best n-gram
             match in source and window around that position.
          2. Otherwise, search for the field-name (last path segment) as a label
             and window around the first 1–2 matches.
        Falls back to `source[:max_chars]` if no anchor lands.
        """
        full = source or ""
        if not full:
            return ""
        if len(full) <= window * 2:
            return full[:max_chars]

        field_name = path.split(".")[-1].split("[")[0]
        positions: List[int] = []
        src_lower = full.lower()

        # Strategy 1: fuzzy-locate the suspect value
        if isinstance(current_value, str) and len(current_value.strip()) >= 3:
            cv = current_value.strip().lower()
            best_pos, best_overlap = -1, 0
            seen_pos: set = set()
            for k in range(0, max(1, len(cv) - 2), max(1, len(cv) // 6)):
                ngram = cv[k:k + 3]
                if len(ngram) < 3 or not ngram.strip():
                    continue
                start = 0
                while True:
                    idx = src_lower.find(ngram, start)
                    if idx < 0:
                        break
                    if idx not in seen_pos:
                        seen_pos.add(idx)
                        # Count overlapping 3-grams of cv near this position
                        window_text = src_lower[max(0, idx - 30): idx + len(cv) + 30]
                        overlap = sum(
                            1 for j in range(0, len(cv) - 2)
                            if cv[j:j + 3] in window_text
                        )
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_pos = idx
                    start = idx + 1
            if best_pos >= 0:
                positions.append(best_pos)

        # Strategy 2: label search by field name
        if not positions and len(field_name) >= 2:
            for pattern in (
                rf"\b{re.escape(field_name)}\s*[:\-]",
                rf"\b{re.escape(field_name.replace('_', ' '))}\s*[:\-]",
                rf"\b{re.escape(field_name)}\b",
            ):
                try:
                    for m in re.finditer(pattern, full, re.IGNORECASE):
                        positions.append(m.start())
                        if len(positions) >= 2:
                            break
                except re.error:
                    continue
                if positions:
                    break

        if not positions:
            return full[:max_chars]

        # Concatenate ≤ 3 windows, deduplicated by coarse 100-char bucket
        seen_bucket: set = set()
        slices: List[str] = []
        for pos in sorted(positions)[:3]:
            start = max(0, pos - window)
            end = min(len(full), pos + window)
            bucket = (start // 100, end // 100)
            if bucket in seen_bucket:
                continue
            seen_bucket.add(bucket)
            slices.append(full[start:end])
        out = "\n…\n".join(slices) if slices else full[:max_chars]
        return out[:max_chars]

    @staticmethod
    def _walk_schema_leaf_strings(
        model_type: Type[BaseModel],
        prefix: str = "",
        _seen: Optional[set] = None,
    ) -> List[tuple]:
        """
        Yield (dotted_path, description) for every leaf-string field in the
        Pydantic schema tree. Walks into nested BaseModels even if the
        record value at that point is null — so wholly-null sub-objects
        still surface their child string fields as candidates.

        Skips: meta fields, list-of-objects (line items are not entity-shaped).
        """
        import typing
        SKIP = {
            "reasoning_thoughts", "confidence_score", "page_references",
            "source_evidence", "domain", "param_type", "is_high_density",
        }
        if _seen is None:
            _seen = set()
        if model_type in _seen or not hasattr(model_type, "model_fields"):
            return []
        _seen.add(model_type)

        results: List[tuple] = []
        for fname, finfo in model_type.model_fields.items():
            if fname in SKIP:
                continue
            ann = finfo.annotation
            origin = typing.get_origin(ann)
            args = typing.get_args(ann)
            # Unwrap Optional / Union[X, None]
            inner = ann
            if origin is typing.Union:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    inner = non_none[0]
                    origin = typing.get_origin(inner)
                    args = typing.get_args(inner)
            full = f"{prefix}.{fname}" if prefix else fname
            # Leaf string
            if inner is str:
                results.append((full, finfo.description))
                continue
            # Nested model
            sub = ExtractorAgent._unwrap_model_class(inner)
            if sub is not None and origin is not list:
                results.extend(ExtractorAgent._walk_schema_leaf_strings(sub, full, _seen))
            # list-of-objects: skip (line items / parameters — not entity leaves)
        return results

    # ------------------------------------------------------------------
    # Targeted-retry core
    # ------------------------------------------------------------------

    @staticmethod
    def _format_record_for_retry(
        record: Any,
        target_path: str,
        max_chars: int = 2000,
    ) -> str:
        """
        Produce a compact JSON view of the current record with the target
        path replaced by the sentinel "<TO_FIND>", so the single-field retry
        prompt can show the model which fields are already filled. Strips
        bulky metadata (reasoning_thoughts, source_evidence, page_references)
        and any list fields longer than 2 items (line items, parameters) to
        keep the prompt small. Purely structural — no schema-specific logic.
        """
        import copy

        SENTINEL = "<TO_FIND>"
        SKIP_KEYS = {
            "reasoning_thoughts", "source_evidence", "page_references",
            "confidence_score",
        }

        def prune(node: Any) -> Any:
            if isinstance(node, dict):
                out = {}
                for k, v in node.items():
                    if k in SKIP_KEYS:
                        continue
                    out[k] = prune(v)
                return out
            if isinstance(node, list):
                if len(node) > 2:
                    return [prune(node[0]), prune(node[1]), f"… +{len(node) - 2} more"]
                return [prune(item) for item in node]
            return node

        try:
            annotated = prune(copy.deepcopy(record))
        except Exception:
            annotated = record

        segs = ExtractorAgent._parse_path_segments(target_path)
        cur = annotated
        for seg in segs[:-1]:
            if isinstance(seg, int):
                if isinstance(cur, list) and 0 <= seg < len(cur):
                    cur = cur[seg]
                else:
                    cur = None
                    break
            else:
                if not isinstance(cur, dict):
                    cur = None
                    break
                if cur.get(seg) is None:
                    cur[seg] = {}
                cur = cur[seg]

        if cur is not None and segs:
            last = segs[-1]
            if isinstance(last, int):
                if isinstance(cur, list) and 0 <= last < len(cur):
                    cur[last] = SENTINEL
            elif isinstance(cur, dict):
                cur[last] = SENTINEL

        try:
            text = json.dumps(annotated, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(annotated)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… (truncated)"
        return text

    @staticmethod
    def _clean_retry_response(text: Optional[str]) -> str:
        """Strip common LLM preambles and quotes; return first non-empty line."""
        if not text:
            return ""
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(
                r"^\s*(VALUE|ANSWER|RESULT|RESPONSE|OUTPUT)\s*[:\-=]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            return line.strip().strip("\"'`").strip()
        return ""

    @staticmethod
    def _verify_retry_response(
        response: str,
        source_markdown: str,
        threshold: float = 0.85,
    ) -> tuple:
        """
        Validate retry response against source. Returns (accepted, final_value, detail).
        - Empty / 'null' / 'none' / 'n/a'  → (False, None, 'model_says_not_present')
        - Verbatim substring of source     → (True, source-cased span, 'verbatim_match')
        - Fuzzy snap with ratio ≥ threshold → (True, snapped, 'fuzzy_snap_<r>')
        - Otherwise                        → (False, None, 'not_in_source_<r>')

        The verbatim path returns the SOURCE's rendering of the span, not the
        model's response — matching is case-insensitive, so the response may
        be case-mangled while the source is ground truth.
        """
        import difflib

        r = (response or "").strip()
        if not r or r.lower() in {"null", "none", "n/a", "na", "-"}:
            return (False, None, "model_says_not_present")

        def normalize(s: str) -> str:
            t = re.sub(r"<!--[^>]*-->", " ", s or "")
            t = re.sub(r"\s+", " ", t)
            return t.lower().strip()

        norm_source = normalize(source_markdown)
        norm_resp = normalize(r)
        if len(norm_resp) < 2 or not norm_source:
            return (False, None, "response_too_short")

        if norm_resp in norm_source:
            # Snap to the source's own casing/whitespace: the match above is
            # case-insensitive, so a case-mangled response still lands here.
            sep = r"(?:\s|<!--[^>]*-->)+"
            pattern = sep.join(re.escape(t) for t in norm_resp.split())
            try:
                m = re.search(pattern, source_markdown or "", re.IGNORECASE)
            except re.error:
                m = None
            if m:
                restored = re.sub(r"<!--[^>]*-->", " ", m.group(0)).strip()
                return (True, restored, "verbatim_match")
            return (True, r, "verbatim_match")

        L = len(norm_resp)
        best_ratio = 0.0
        best_span = None
        for wl in range(max(3, L - 3), L + 4):
            if wl > len(norm_source):
                continue
            for i in range(0, len(norm_source) - wl + 1):
                cand = norm_source[i:i + wl]
                if not any(
                    norm_resp[k:k + 3] in cand
                    for k in range(0, max(1, len(norm_resp) - 2), 2)
                ):
                    continue
                ratio = difflib.SequenceMatcher(None, norm_resp, cand).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (i, wl)
                    if ratio == 1.0:
                        break
            if best_ratio == 1.0:
                break

        if best_span and best_ratio >= threshold:
            i, wl = best_span
            cand_lower = norm_source[i:i + wl]
            m = re.search(re.escape(cand_lower), source_markdown or "", re.IGNORECASE)
            snapped = (m.group(0).strip() if m else cand_lower)
            return (True, snapped, f"fuzzy_snap_{best_ratio:.2f}")

        # Digit-aware fallback: if the response is digit-heavy (dates, IDs,
        # codes, phone numbers, etc.), compare digit-strip forms against
        # contiguous digit-rich spans in source. Catches separator drops
        # ("20046" → "2020/4/6") that the verbatim/fuzzy passes miss.
        non_ws = [c for c in r if not c.isspace()]
        if non_ws:
            digit_or_sep = sum(1 for c in non_ws if c.isdigit() or c in "-/.,:")
            if digit_or_sep / len(non_ws) >= 0.6:
                resp_digits = re.sub(r"\D+", "", r)
                if len(resp_digits) >= 3:
                    best_dr = 0.0
                    best_span_text = None
                    for sm in re.finditer(r"[\d][\d/\-\.\s,:]{2,}[\d]", source_markdown or ""):
                        span = sm.group(0).strip()
                        span_digits = re.sub(r"\D+", "", span)
                        if not span_digits:
                            continue
                        dr = difflib.SequenceMatcher(None, resp_digits, span_digits).ratio()
                        if dr > best_dr:
                            best_dr = dr
                            best_span_text = span
                            if dr == 1.0:
                                break
                    if best_span_text and best_dr >= threshold:
                        return (True, best_span_text, f"digit_snap_{best_dr:.2f}")

        return (False, None, f"not_in_source_best_ratio_{best_ratio:.2f}")

    def _build_retry_prompt(
        self,
        path: str,
        current_value: Any,
        description: Optional[str],
        type_hint: Optional[str],
        source: str,
        record_snapshot: Optional[str] = None,
    ) -> str:
        field_name = path.split(".")[-1].split("[")[0]
        parent = ".".join(path.split(".")[:-1]) or "<root>"
        desc_line = f"DESCRIPTION:   {description}\n" if description else ""
        type_line = f"TYPE:          {type_hint}\n" if type_hint else ""
        if current_value is None or (isinstance(current_value, str) and not current_value.strip()):
            suspect_line = (
                "CURRENT VALUE: null\n"
                "HINT:          The previous extraction left this field empty. "
                "Find it in the source if present (it may be near a label like "
                f"`{field_name.replace('_', ' ').title()}:` or similar)."
            )
        else:
            suspect_line = (
                f"SUSPECT VALUE: {current_value}\n"
                "HINT:          The previous extraction is suspected wrong — "
                "likely a character-level corruption of a value present in the "
                "source. Locate a value of the same SHAPE near the suspect "
                "value's likely position and copy it verbatim."
            )
        record_block = ""
        if record_snapshot:
            record_block = (
                "\nCURRENT RECORD (the field you must find is marked `<TO_FIND>` — "
                "use the rest of the record as context so you don't duplicate a "
                "neighbouring field's value):\n"
                f"{record_snapshot}\n"
            )
        return f"""You are a forensic data extractor. Find ONE specific value in the source text.

FIELD PATH:    {path}
FIELD NAME:    {field_name}
PARENT:        {parent}
{type_line}{desc_line}{suspect_line}
{record_block}
SOURCE (windowed around the likely position):
{source}

RULES:
1. COPY THE VALUE VERBATIM from the source. No paraphrasing, no normalisation, no quotes.
2. Reply with ONLY the value on a single line. No JSON, no preamble.
3. If the value truly does not appear in the source, reply with exactly: null
4. Do not repeat a value that already appears for another field in the CURRENT RECORD above — if you would, prefer `null`.

VALUE:
"""

    def _retry_flagged_and_null_strings(
        self,
        record: Dict[str, Any],
        schema_model: Type[BaseModel],
        source_markdown: str,
        grounding_stats: Dict[str, Any],
        max_retries: int = 5,
        threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Post-hoc targeted span extraction.

        Two candidate sources, in order:
          1. `grounding_stats["flagged"]` — values the verifier proved are
             not in the source (high-confidence repair targets).
          2. Schema-driven null leaf strings — every leaf-string field in
             the synthesized schema whose value at that path is null/empty.
             Walks the Pydantic model tree, not the record, so wholly-null
             sub-objects still surface their child fields.

        Each candidate triggers one LLM call asking for the verbatim source
        value. Responses must pass `_verify_retry_response` against the
        source (verbatim or fuzzy ≥ threshold) — otherwise the original
        value is left untouched.

        Mutates `record` in place for accepted retries.
        Returns: list of audit entries.
        """
        if not source_markdown:
            return []

        candidates: List[Dict[str, Any]] = []
        seen_paths: set = set()

        # 1a. Flagged values (verifier-proven wrong) — highest priority
        for f in grounding_stats.get("flagged", []) or []:
            if f.get("kind") == "numeric":
                # Numeric flags are the numeric verifier's domain; a string
                # retry answer would type-clash with the schema field.
                continue
            path = f.get("path")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            candidates.append({
                "path": path,
                "current_value": f.get("value"),
                "reason": "flagged",
                "description": self._description_for_path(schema_model, path),
                "type_hint": self._type_hint_for_path(schema_model, path),
            })

        # 1b. Null leaf strings — walk the SCHEMA so wholly-null parents drill in.
        # Interleave by top-level container so cap doesn't burn all on one side
        # (e.g., `from.*` swallowing budget needed for `to.*`).
        if len(candidates) < max_retries:
            by_container: Dict[str, List[tuple]] = {}
            order: List[str] = []
            for path, description in self._walk_schema_leaf_strings(schema_model):
                if path in seen_paths:
                    continue
                current = self._get_at_path(record, path)
                if current is None or (isinstance(current, str) and not current.strip()):
                    parent = ".".join(path.split(".")[:-1]) or "<root>"
                    if parent not in by_container:
                        by_container[parent] = []
                        order.append(parent)
                    by_container[parent].append((path, description))

            # Round-robin across containers
            while by_container and len(candidates) < max_retries * 2:
                progressed = False
                for parent in list(order):
                    bucket = by_container.get(parent)
                    if not bucket:
                        continue
                    path, description = bucket.pop(0)
                    candidates.append({
                        "path": path,
                        "current_value": None,
                        "reason": "null_leaf",
                        "description": description,
                        "type_hint": self._type_hint_for_path(schema_model, path),
                    })
                    seen_paths.add(path)
                    if not bucket:
                        del by_container[parent]
                    progressed = True
                    if len(candidates) >= max_retries * 2:
                        break
                if not progressed:
                    break

        candidates = candidates[:max_retries]
        if not candidates:
            return []

        audit: List[Dict[str, Any]] = []

        for cand in candidates:
            prompt_source = self._anchored_source(
                source=source_markdown,
                path=cand["path"],
                current_value=cand["current_value"],
            )
            record_snapshot = self._format_record_for_retry(record, cand["path"])
            prompt = self._build_retry_prompt(
                path=cand["path"],
                current_value=cand["current_value"],
                description=cand.get("description"),
                type_hint=cand.get("type_hint"),
                source=prompt_source,
                record_snapshot=record_snapshot,
            )
            try:
                raw = self.client.generate(image=None, prompt=prompt)
            except Exception as e:
                audit.append({
                    "path": cand["path"],
                    "reason": cand["reason"],
                    "before": cand["current_value"],
                    "retry_response": None,
                    "after": cand["current_value"],
                    "accepted": False,
                    "reason_detail": f"llm_error: {e}",
                })
                continue

            cleaned = self._clean_retry_response(raw)
            accepted, final_value, detail = self._verify_retry_response(
                cleaned, source_markdown, threshold
            )

            entry = {
                "path": cand["path"],
                "reason": cand["reason"],
                "before": cand["current_value"],
                "retry_response": cleaned,
                "after": cand["current_value"] if not accepted else final_value,
                "accepted": accepted,
                "reason_detail": detail,
            }
            if accepted:
                ok = self._set_at_path(record, cand["path"], final_value)
                if not ok:
                    entry["accepted"] = False
                    entry["after"] = cand["current_value"]
                    entry["reason_detail"] = "set_at_path_failed"

            audit.append(entry)

        return audit

    def extract_structured(
        self,
        image: Optional[Image.Image],
        prompt: str,
        response_model: Type[T],
        domain: str = "General",
        is_high_density: bool = False,
        context_markdown: Optional[str] = None,
        context_nodes: Optional[List[HierarchicalNode]] = None,
        target_schema_name: Optional[str] = None,
        target_schema_json: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        use_grounding: bool = False,
    ) -> T:
        """
        Extracts structured data using the 'Markdown Harvest' (Two-Pass) architecture.
        Pass 1: Harvest raw technical data as Markdown to bypass JSON token limits.
        Pass 2: Refine consolidated Markdown into the final Pydantic schema.
        """
        # Decide strategy: High-density domains use direct structured extraction
        # This replaces the broken Two-Pass harvest→refine which hallucinated
        # for text-only providers on complex documents.
        
        if not is_high_density and len(context_markdown or "") < 4000:
            logger.info(f"Librarian Agent: Direct structured extraction for {domain}")
            
            # Concatenate ALL graph content (no fragmentation)
            if context_nodes:
                full_context = "\n\n".join(n.content for n in context_nodes if n.content)
            else:
                full_context = context_markdown or ""
            full_context = full_context[:12000]  # Fit in model context window
            
            hint = self._build_schema_aligned_hint(response_model, domain)
            direct_prompt = f"""{hint}

SOURCE CONTENT:
{full_context}
"""
            result = self.client.generate_structured(
                image=None,
                prompt=direct_prompt,
                response_model=response_model,
                max_tokens=4096,
                trace_dir=(trace_context or {}).get("trace_dir"),
                trace_key="final_refinement",
                target_schema_name=target_schema_name,
            )
            if result:
                # Check if it's non-empty
                excluded = {'reasoning_thoughts', 'confidence_score', 'page_references'}
                result_dump = result.model_dump(exclude=excluded)
                default_dump = response_model.model_construct().model_dump(exclude=excluded)
                if result_dump != default_dump:
                    # Route the single-shot result through the SAME finalization
                    # (grounding verification, retry, provenance) the batched
                    # path uses — so low-density docs are checked for fabrication
                    # too, instead of returning unverified.
                    return self._finalize_record(
                        result,
                        response_model=response_model,
                        context_markdown=context_markdown,
                        target_schema_json=target_schema_json,
                        target_schema_name=target_schema_name,
                        trace_context=trace_context,
                        use_grounding=use_grounding,
                        total_batches=1,
                        batch_success_count=1,
                    )
                else:
                    logger.warning("Direct extraction returned empty. Falling through to One-Pass batched mode.")
            else:
                logger.warning("Direct extraction failed. Falling through to One-Pass batched mode.")

        # Standard One-Pass logic for simple domains...
        hint = self._specialist_hint(domain)
        
        # Prefer node-aware semantic batches to avoid splitting tables across boundaries.
        if context_nodes:
            semantic_batches = GraphBuilder.to_extraction_batches(
                context_nodes,
                max_chars=8000,  # Large batches to keep tables intact
                preserve_atomic=False,  # Shatter Mode: Break through the truncation ceiling
            )
            batches = [b.get("text", "") for b in semantic_batches if b.get("text")]
        else:
            graph_text = context_markdown if context_markdown else "No graph context provided."
            max_batch_size = 18000  # Safe chars per batch for 16k token window
            batches = [graph_text[i:i+max_batch_size] for i in range(0, len(graph_text), max_batch_size)]

        if not batches:
            batches = [""]

        batch_results = []
        batch_success_count = 0
        quality_rollup: List[Dict[str, Any]] = []
        logger.info(f"Librarian Agent: Starting {domain} extraction using {response_model.__name__} in {len(batches)} batches.")

        for i, batch_content in enumerate(batches):
            batch_id = f"batch_{i+1:04d}"
            # Schema-aligned hint: instructions are derived from the ACTUAL
            # schema fields. The domain-specific SPECIALIST_HINTS (added below)
            # describe *what kind of content* to look for, not specific field
            # names — that stays in the schema-driven hint.
            schema_hint = self._build_schema_aligned_hint(response_model, domain)
            domain_hint = self._specialist_hint(domain)
            batch_prompt = f"""{schema_hint}

### DOMAIN GUIDANCE ({domain}):
{domain_hint}

SOURCE CONTENT (Segment {i+1}/{len(batches)}):
{batch_content}
"""
            
            result = self.client.generate_structured(
                image=image,
                prompt=batch_prompt,
                response_model=response_model,
                max_tokens=4096,  # Stable hardware-aligned limit
                trace_dir=(trace_context or {}).get("trace_dir"),
                trace_key=batch_id,
                target_schema_name=target_schema_name,
            )
            
            if result:
                # Automatic Success Detection: Did the model populate anything non-default?
                populated = False
                default_instance = response_model.model_construct()
                excluded_fields = {'reasoning_thoughts', 'confidence_score', 'page_references'}
                
                result_dump = result.model_dump(exclude=excluded_fields)
                default_dump = default_instance.model_dump(exclude=excluded_fields)
                
                if result_dump != default_dump:
                    populated = True
                
                if populated:
                    batch_results.append(result)
                    batch_success_count += 1
                    quality = self._score_quality(result, response_model)
                    quality_rollup.append(quality)
                    logger.info(f"    Batch {i+1}: Extraction SUCCESS. Fields populated.")
                    self._append_jsonl(
                        (trace_context or {}).get("batch_trace_file"),
                        {
                            "batch_id": batch_id,
                            "schema": target_schema_name or response_model.__name__,
                            "input_chars": len(batch_content),
                            "parse_status": "success",
                            "populated_fields_count": quality["populated_fields_count"],
                            "required_fields_missing": quality["required_fields_missing"],
                        },
                    )
                else:
                    logger.debug(f"    Batch {i+1}: Empty - no relevant data found.")
                    self._append_jsonl(
                        (trace_context or {}).get("batch_trace_file"),
                        {
                            "batch_id": batch_id,
                            "schema": target_schema_name or response_model.__name__,
                            "input_chars": len(batch_content),
                            "parse_status": "empty",
                        },
                    )
            else:
                logger.warning(f"    Batch {i+1}: Extraction failed.")
                self._append_jsonl(
                    (trace_context or {}).get("batch_trace_file"),
                    {
                        "batch_id": batch_id,
                        "schema": target_schema_name or response_model.__name__,
                        "input_chars": len(batch_content),
                        "parse_status": "failed",
                    },
                )

        if not batch_results:
            # CHECKPOINT: Fallback mechanism for synthesized models
            # If we used a synthesized model and it failed completely, retry with Universal Schema
            is_synthesized = response_model.__name__ == "LibrarianUniversalHardware" and response_model != LibrarianUniversalHardware
            
            if is_synthesized:
                logger.warning("Librarian Agent: Synthesized model extraction failed. Falling back to Universal Hardware schema...")
                return self.extract_structured(
                    image=image,
                    prompt=prompt,
                    response_model=LibrarianUniversalHardware,
                    domain=domain,
                    context_markdown=context_markdown,
                    use_grounding=use_grounding,
                    trace_context=trace_context,
                    target_schema_name="LibrarianUniversalHardware"
                )

            details = {
                "status": "failed",
                "reason": "no_data_extracted",
                "schema": target_schema_name or response_model.__name__,
                "batch_success_count": batch_success_count,
                "total_batches": len(batches),
                "target_schema_json": target_schema_json,
            }
            logger.error("Librarian Agent: No data extracted from any batch!")
            self._append_jsonl((trace_context or {}).get("parse_failures_file"), details)
            self._write_json(
                (trace_context or {}).get("validation_summary_file"),
                {
                    "status": "failed",
                    "schema": target_schema_name or response_model.__name__,
                    "batch_success_count": batch_success_count,
                    "total_batches": len(batches),
                    "populated_fields_count": 0,
                    "required_fields_missing": [],
                    "reason": "no_data_extracted",
                },
            )
            raise ExtractionFailureError("No data extracted from any batch", details)

        # Synthesis pass to combine batch results into one master record
        master = self._synthesize_results(batch_results, response_model, domain)

        return self._finalize_record(
            master,
            response_model=response_model,
            context_markdown=context_markdown,
            target_schema_json=target_schema_json,
            target_schema_name=target_schema_name,
            trace_context=trace_context,
            use_grounding=use_grounding,
            total_batches=len(batches),
            batch_success_count=batch_success_count,
        )

    def _finalize_record(
        self,
        master: T,
        *,
        response_model: Type[T],
        context_markdown: Optional[str],
        target_schema_json: Optional[Dict[str, Any]],
        target_schema_name: Optional[str],
        trace_context: Optional[Dict[str, Any]],
        use_grounding: bool,
        total_batches: int,
        batch_success_count: int,
    ) -> T:
        """Shared finalization for BOTH extraction paths (direct + batched).

        Runs projection, currency inference, span-grounding verification,
        flagged-value retry, provenance auto-fill, and sets
        self._last_grounding_stats. Centralizing this guarantees the single-shot
        direct path is verified identically to the batched path — it previously
        returned early and skipped grounding, leaving low-density docs unchecked
        for fabrication.
        """
        # Grounding pass for high-precision evidence (optional)
        if use_grounding:
            self._ground_block_evidence(master, context_markdown or "")

        # Projection pass: reshape the model output into the target schema's
        # exact field names (structural only — no raw-document content
        # guessing). Missing fields are left empty here and recovered, if at
        # all, by the source-grounded retry below.
        master_dict = master.model_dump()
        refined_dict = self.project_to_schema(
            source_payload=master_dict,
            target_schema_json=target_schema_json or response_model.model_json_schema(),
            context_markdown=context_markdown or ""
        )

        # Currency-aware post-extraction normalization
        currency_filled = self._infer_currency_inplace(refined_dict)
        if currency_filled:
            logger.info(f"      [Currency] Inferred {currency_filled} currency code(s) from value symbols.")

        # Verbatim span-grounding: repair character-level extractor defects
        grounding_stats = self._verify_string_spans(refined_dict, context_markdown or "")
        for rep in grounding_stats["repaired"]:
            logger.info(
                f"      [Grounding] {rep['path']} repaired "
                f"({rep['ratio']:.2f}): '{rep['before']}' → '{rep['after']}'"
            )
        if grounding_stats["flagged"]:
            logger.info(
                f"      [Grounding] {len(grounding_stats['flagged'])} field(s) "
                f"not found in source — flagged for manual review."
            )

        # Targeted post-hoc retry: ask the model verbatim for each flagged
        # value and each null leaf-string in the synthesized schema. The
        # source-verification gate refuses invented values; cap=3 bounds
        # latency. Original values are preserved when retry doesn't pass.
        retry_audit = self._retry_flagged_and_null_strings(
            record=refined_dict,
            schema_model=response_model,
            source_markdown=context_markdown or "",
            grounding_stats=grounding_stats,
            max_retries=5,
        )
        accepted_retries = [r for r in retry_audit if r["accepted"]]
        for r in accepted_retries:
            logger.info(
                f"      [Retry] {r['path']} ({r['reason']}): "
                f"{r['before']!r} → {r['after']!r} [{r['reason_detail']}]"
            )
        for r in retry_audit:
            if not r["accepted"]:
                logger.debug(
                    f"      [Retry] {r['path']} ({r['reason']}): "
                    f"rejected — {r['reason_detail']}"
                )
        # Recall coverage: detect source tables the extraction mostly missed
        # (e.g. items:[] plus a rationalization) and re-extract empty list
        # fields from the missed table. The precision checks above can only
        # judge values that are present — this is the omission check.
        coverage = self._audit_list_coverage(
            refined_dict, response_model, context_markdown or "", trace_context
        )
        coverage_recovered = bool(
            coverage and any(r.get("accepted") for r in coverage.get("recovered", []))
        )

        if accepted_retries or coverage_recovered:
            # Refresh provenance now that previously-empty/wrong fields are
            # filled — but carry over the first pass's repair audit: the fresh
            # pass sees already-repaired values and would report repaired=0,
            # hiding the repairs from the final _grounding block. The refresh
            # also case-restores and span-verifies coverage-recovered items.
            prior_repairs = grounding_stats["repaired"]
            grounding_stats = self._verify_string_spans(refined_dict, context_markdown or "")
            grounding_stats["repaired"] = prior_repairs + grounding_stats["repaired"]
        if coverage:
            grounding_stats["coverage"] = coverage
        grounding_stats["retries"] = retry_audit
        grounding_stats["retries_attempted"] = len(retry_audit)
        grounding_stats["retries_accepted"] = len(accepted_retries)

        # Auto-fill page_references / source_evidence from grounding provenance.
        # We only fill empty fields — the LLM's own answers take priority.
        provenance = grounding_stats.get("provenance", [])
        if provenance and isinstance(refined_dict, dict):
            pages = sorted({
                p["page"] for p in provenance
                if isinstance(p.get("page"), int)
            })
            if "page_references" in refined_dict and not refined_dict.get("page_references") and pages:
                refined_dict["page_references"] = pages[:10]
                logger.info(f"      [Grounding] Auto-filled page_references={pages[:10]} from {len(provenance)} verified spans.")
            if "source_evidence" in refined_dict and not refined_dict.get("source_evidence"):
                # Take the top-ratio entries, dedupe by snippet (case-insensitive),
                # cap at 6 to keep the record compact.
                seen: set = set()
                evidence: List[Dict[str, Any]] = []
                for p in sorted(provenance, key=lambda x: -x.get("ratio", 0)):
                    snip = (p.get("snippet") or "").strip()
                    key = snip.lower()
                    if not snip or key in seen:
                        continue
                    seen.add(key)
                    evidence.append({
                        "text_snippet": snip[:200],
                        "page_number": p.get("page"),
                        "confidence": p.get("ratio", 1.0),
                    })
                    if len(evidence) >= 6:
                        break
                if evidence:
                    refined_dict["source_evidence"] = evidence
                    logger.info(f"      [Grounding] Auto-filled source_evidence with {len(evidence)} item(s).")

        self._last_grounding_stats = grounding_stats

        # Scrub PydanticUndefined sentinels — these can leak from sub-models
        # built via model_construct() during batch merging (when a leaf field
        # has no default and wasn't populated). Pydantic 2 refuses to validate
        # a dict containing them; coerce to None so Optional fields pass.
        from pydantic_core import PydanticUndefined as _PU
        def _scrub(o):
            if o is _PU:
                return None
            if isinstance(o, dict):
                return {k: _scrub(v) for k, v in o.items() if v is not _PU}
            if isinstance(o, list):
                return [_scrub(v) for v in o if v is not _PU]
            return o
        refined_dict = _scrub(refined_dict)

        # Convert back to response_model
        master = response_model.model_validate(refined_dict)

        final_quality = self._score_quality(master, response_model)
        final_quality["batch_success_count"] = batch_success_count
        self._write_json(
            (trace_context or {}).get("validation_summary_file"),
            {
                "status": "success",
                "schema": target_schema_name or response_model.__name__,
                "batch_success_count": batch_success_count,
                "total_batches": total_batches,
                "populated_fields_count": final_quality["populated_fields_count"],
                "required_fields_missing": final_quality["required_fields_missing"],
            },
        )
        return master


    def _synthesize_results(self, results: List[T], response_model: Type[T], domain: str) -> T:
        """Merges multiple partial results into a single coherent 'Master Record' using additive logic."""
        master = response_model()
        all_reasoning = []
        
        logger.info(f"    Additive Synthesis: Merging {len(results)} partial results...")
        
        # Determine all available blocks/fields in the response model
        model_fields = response_model.model_fields.keys()
        
        for r in results:
            for field in model_fields:
                r_val = getattr(r, field, None)
                if r_val is None:
                    continue
                
                m_val = getattr(master, field, None)
                
                # Case 1: The field is a list (e.g., parameters, connectors, leds, standards, entities, timeline)
                if isinstance(r_val, list):
                    if m_val is None:
                        setattr(master, field, [])
                        m_val = getattr(master, field)
                    # Extend and avoid duplicates if appropriate
                    if field in ('page_references', 'standards', 'key_points'):
                        for item in r_val:
                            if item not in m_val:
                                m_val.append(item)
                    else:
                        m_val.extend(r_val)
                
                # Case 2: The field is a sub-model (a Module/Block)
                elif isinstance(r_val, BaseModel):
                    if m_val is None:
                        setattr(master, field, r_val)
                    else:
                        # Recursive merge for block contents
                        for sub_field in r_val.model_fields:
                            sub_r_val = getattr(r_val, sub_field, None)
                            sub_m_val = getattr(m_val, sub_field, None)
                            if self._is_non_empty(sub_r_val) and not self._is_non_empty(sub_m_val):
                                setattr(m_val, sub_field, sub_r_val)
                            elif isinstance(sub_r_val, list) and isinstance(sub_m_val, list):
                                sub_m_val.extend(sub_r_val)
                
                # Case 3: Scalar fields
                elif not self._is_non_empty(m_val):
                    setattr(master, field, r_val)

            # Special handling for reasoning/thought accumulation
            if hasattr(r, 'reasoning_thoughts') and r.reasoning_thoughts:
                all_reasoning.append(r.reasoning_thoughts)
            elif hasattr(r, 'summary') and r.summary: 
                all_reasoning.append(r.summary)

        # 2b. Deduplicate parameter-shaped lists by name, wherever they appear.
        # A list is "parameter-shaped" if its items expose name/value/unit (the
        # TechParameter shape). We detect this STRUCTURALLY rather than keying on
        # a field literally named 'parameters', so the pass stays domain-agnostic
        # and never touches differently-shaped lists like line_items or entities.
        def _is_param_shaped(lst) -> bool:
            if not isinstance(lst, list) or not lst:
                return False
            item = next((x for x in lst if x is not None), None)
            if isinstance(item, BaseModel):
                keys = set(item.model_fields.keys())
            elif isinstance(item, dict):
                keys = set(item.keys())
            else:
                return False
            return {'name', 'value', 'unit'} <= keys

        def _dedupe_param_list(params: list) -> list:
            seen = set()
            deduped = []
            for p in params:
                # Smart unit splitting: if unit is null but value carries it, split it out.
                p_name = getattr(p, 'name', None) or (p.get('name') if isinstance(p, dict) else None)
                p_val = getattr(p, 'value', None) or (p.get('value') if isinstance(p, dict) else None)
                p_unit = getattr(p, 'unit', None) or (p.get('unit') if isinstance(p, dict) else None)

                if p_val and not self._is_non_empty(p_unit):
                    # Aggressive Cleaning: Remove non-printable/weird spaces
                    clean_val = str(p_val).replace('\xa0', ' ').strip()
                    # Greedy regex for Value + Unit
                    m = re.search(r"^([\d\.,\-\+\s/]+)\s*([a-zA-Z°Ω%µ\d]*[a-zA-Z°Ω%µ][²³]?)$", clean_val)
                    if m:
                        new_val, new_unit = m.group(1).strip(), m.group(2).strip()
                        # Validation: Unit should look like a unit (not a trailing number)
                        if new_unit and not new_unit.isdigit():
                            logger.info(f"      [Heuristic] Split unit '{new_unit}' from value '{new_val}' for {p_name}")
                            if isinstance(p, dict):
                                p['value'], p['unit'] = new_val, new_unit
                            else:
                                setattr(p, 'value', new_val)
                                setattr(p, 'unit', new_unit)

                if p_name and p_name in seen:
                    continue
                if p_name:
                    seen.add(p_name)
                deduped.append(p)
            return deduped

        for field in model_fields:
            m_val = getattr(master, field, None)
            # Top-level param-shaped list
            if _is_param_shaped(m_val):
                setattr(master, field, _dedupe_param_list(m_val))
            # Param-shaped list nested one level inside a sub-model block
            elif isinstance(m_val, BaseModel):
                for sub_field in m_val.model_fields:
                    sub_val = getattr(m_val, sub_field, None)
                    if _is_param_shaped(sub_val):
                        setattr(m_val, sub_field, _dedupe_param_list(sub_val))



        # 4. Use VLM to generate a cohesive SUMMARY from the collected findings.
        # default=str in json.dumps below makes the dump robust against
        # PydanticUndefined sentinels that can leak when a sub-model was
        # constructed via model_construct() during batch merging without
        # touching every leaf field. (mode="json" can't be used here because
        # pydantic 2's own json encoder also refuses PydanticUndefined.)
        combined_json = master.model_dump()
        distill_prompt = f"""You are a High-Precision Librarian Specialist.
Your goal is to extract structured data into a {response_model.__name__} schema.

STRICT RULES:
1. Respond ONLY with a structural JSON object.
2. DO NOT include any introductory text, summaries, or conversational filler.
3. If a field is not found in the source, use null or an empty list [].
4. Focus on high precision. Only extract what is explicitly stated or clearly implied.

SPECIALIST HINTS:
{self._specialist_hint(domain)}

SOURCE CONTENT:
{json.dumps(combined_json, indent=2, default=str)}

### COLLECTED BATCH THOUGHTS:
{chr(10).join(all_reasoning)}
"""
        vlm_synthesis = self.client.generate_structured(
            image=None,
            prompt=distill_prompt,
            response_model=response_model
        )
        
        if vlm_synthesis:
            # Update narrative fields without overwriting structured lists that were already merged
            for field in ('general_info', 'summary', 'purpose', 'reasoning_thoughts'):
                synth_val = getattr(vlm_synthesis, field, None)
                if synth_val:
                    setattr(master, field, synth_val)
            
            # Check for standards if applicable
            if hasattr(vlm_synthesis, 'standards') and vlm_synthesis.standards:
                existing = set(getattr(master, 'standards', []) or [])
                for s in vlm_synthesis.standards:
                    if s and s not in existing:
                        master.standards.append(s)
                        existing.add(s)
        else:
            logger.warning("VLM Distillation failed. Falling back to joined reasoning.")
            master.reasoning_thoughts = "\n---\n".join(all_reasoning)
        
        return master

    def _ground_block_evidence(self, master: T, context_markdown: str) -> None:
        """Attach real ``SourceEvidence`` to every sub-model that declares it.

        Schema-agnostic: any nested block (direct field or list item) with a
        ``source_evidence`` field is groundable, whatever the domain — this
        replaces the old hardcoded block whitelist (killer-#3) AND the
        langextract stub behind it, which fired a doomed LLM call per block
        (``examples=[]`` is rejected) and then stamped a fabricated
        ``context[:200]`` snippet with ``page_number=1, confidence=0.95``
        over the field. Evidence now comes from the same deterministic span
        verifier used for record-level grounding: the block's own values are
        located in the source, and the best-matching real span becomes the
        snippet with its actual page number and match ratio. No LLM call.
        """
        if not context_markdown:
            return

        blocks: List[BaseModel] = []

        def is_groundable(obj: Any) -> bool:
            return isinstance(obj, BaseModel) and "source_evidence" in type(obj).model_fields

        for name in type(master).model_fields:
            value = getattr(master, name, None)
            if is_groundable(value):
                blocks.append(value)
            elif isinstance(value, list):
                blocks.extend(item for item in value if is_groundable(item))

        if not blocks:
            return
        logger.info(f"      [Grounding] Locating source evidence for {len(blocks)} block(s)...")
        for block in blocks:
            self._ground_block(block, context_markdown)

    def _ground_block(self, block: BaseModel, context: str) -> None:
        """Fill ``block.source_evidence`` with the best real span for its values.

        Never overwrites evidence the model itself produced — the LLM's own
        citation takes priority; this only recovers blocks it left blank.
        """
        if getattr(block, "source_evidence", None):
            return

        payload = block.model_dump(exclude={"source_evidence"})
        stats = self._verify_string_spans(payload, context)
        provenance = stats.get("provenance", [])
        if not provenance:
            return

        best = max(
            provenance,
            key=lambda p: (p.get("ratio", 0), len(p.get("snippet") or "")),
        )
        evidence = SourceEvidence(
            text_snippet=best.get("snippet"),
            page_number=best.get("page"),
            confidence=best.get("ratio", 1.0),
        )

        # Match the declared field shape (Optional[SourceEvidence] in the
        # built-in schemas; synthesized schemas may declare a list).
        annotation = type(block).model_fields["source_evidence"].annotation
        wants_list = list in {get_origin(a) for a in (*get_args(annotation), annotation)}
        block.source_evidence = [evidence] if wants_list else evidence

    def _harvest_as_markdown(
        self,
        image: Optional[Image.Image],
        context_nodes: Optional[List[HierarchicalNode]],
        context_markdown: Optional[str],
        domain: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Pass 1: Extract everything as raw Markdown to avoid JSON truncation."""
        if context_nodes:
            semantic_batches = GraphBuilder.to_extraction_batches(
                context_nodes,
                max_chars=3000,
                preserve_atomic=False,
            )
            batches = [b.get("text", "") for b in semantic_batches if b.get("text")]
        else:
            graph_text = context_markdown or "No context provided."
            batches = [graph_text[i:i+3000] for i in range(0, len(graph_text), 3000)]

        harvest_shards = []
        logger.info(f"    Harvest Phase: Processing {len(batches)} visual batches...")

        for i, batch_content in enumerate(batches):
            batch_id = f"harvest_batch_{i+1:04d}"
            harvest_prompt = f"""
I am the Visual Librarian. My goal is to HARVEST every detail from this document.

### SOURCE CONTENT (Segment {i+1}/{len(batches)}):
{batch_content}

### TASK:
Construct a complete, exhaustive Markdown representation of all data found in this segment.
1. Extract EVERY row from EVERY table exactly as it appears.
2. Extract ALL key entities, values, and contextual information relevant to a '{domain}' document.
3. Be EXTREMELY PRECISE. Do not summarize; extract exact values and maintain relationships.
4. Respond ONLY with Markdown text. NO JSON, NO introductory text.
"""
            # For high-density documents, we disable the image to prevent visual hallucinations 
            # and force the model to strictly follow the Graph text nodes.
            raw_text = self.client.generate(
                image=None,
                prompt=harvest_prompt,
                trace_dir=(trace_context or {}).get("trace_dir"),
                trace_key=batch_id,
            )
            if raw_text and len(raw_text.strip()) > 10:
                # Basic line-level deduplication to prevent model repetition loops
                lines = raw_text.strip().split('\n')
                seen = set()
                deduped_lines = []
                for line in lines:
                    clean = line.strip()
                    if clean and clean not in seen:
                        deduped_lines.append(line)
                        seen.add(clean)
                harvest_shards.append('\n'.join(deduped_lines))

        full_harvest = "\n\n---\n\n".join(harvest_shards)
        
        # Save full buffer for debugging
        if trace_context and trace_context.get("trace_dir"):
            harvest_trace_path = Path(trace_context["trace_dir"]) / "harvest_full_buffer.md"
            with open(harvest_trace_path, "w", encoding="utf-8") as f:
                f.write(full_harvest)
                
        return full_harvest

    def _refine_to_structured(
        self,
        markdown_text: str,
        response_model: Type[T],
        domain: str,
        trace_context: Optional[Dict[str, Any]] = None,
        target_schema_name: Optional[str] = None,
    ) -> T:
        """Pass 2: Map the consolidated Markdown text into the structured Pydantic schema."""
        logger.info("    Refinement Phase: Mapping Markdown to Pydantic...")
        
        # We use a conservative batching logic for refinement to ensure high precision
        refine_prompt = f"""
You are a HIGH-FIDELITY DATA REFINER. Your task is to map RAW harvested Markdown into a structured Pydantic schema.

### INPUT DATA (Harvested Markdown):
{markdown_text}

### INSTRUCTIONS:
1. **FULL COVERAGE**: You MUST extract all requested fields based on the schema definition.
2. **NO TRUNCATION**: The Markdown is long. Map every single relevant data point found.
3. **DOMAIN**: This is a {domain} document.
{self._specialist_hint(domain)}

### OUTPUT FORMAT:
Output ONLY a valid JSON object matching the {target_schema_name} schema.
No preamble, no markdown code blocks, JUST the JSON.
"""
        return self.client.generate_structured(
            image=None,  # Pass 2 is text-centric
            prompt=refine_prompt,
            response_model=response_model,
            max_tokens=4096,
            trace_dir=(trace_context or {}).get("trace_dir"),
            trace_key="final_refinement",
            target_schema_name=target_schema_name
        ) or response_model()

    def extract_from_graph(self, nodes: List[HierarchicalNode], target_schema: Dict[str, Any]) -> ExtractionResult:
        """Old entry point, now wraps the new structured logic."""
        batch_context = "\n\n".join([n.content for n in nodes])
        hkg_result = self.extract_structured(None, "Perform industrial hardware extraction.", LibrarianUniversalHardware, domain="Hardware", context_markdown=batch_context)
        
        legacy_data = []
        if hasattr(hkg_result, 'identity') and hkg_result.identity:
            identity_data = hkg_result.identity.model_dump()
            legacy_data.append(Extraction(
                extraction_class="Identity", 
                extraction_text=identity_data.get('product_name', "Product"),
                attributes=identity_data
            ))
        
        return ExtractionResult(
            schema_title="LibrarianUniversalHardware",
            data=legacy_data,
            confidence_score=1.0,
        )


