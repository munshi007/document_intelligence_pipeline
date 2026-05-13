import logging
import json
import re
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, create_model, Field, AliasChoices
from common.vlm_client import VLMClient
from extractor.schema_definitions import (
    IdentificationBlock,
    TechParameter,
    PinAssignment,
    ConnectorSpec,
    LEDBehavior,
    InvoiceHeaderBlock,
    InvoiceLinesBlock,
    TotalsBlock,
    GeneralInfoBlock,
    SourceEvidence,
)


def _match_typed_block(items_schema: Dict[str, Any]) -> Optional[Type[BaseModel]]:
    """
    If an LLM-proposed array-of-object schema matches a known typed block,
    return that Pydantic class. Otherwise None.

    Match policy (most-specific first):
      - PinAssignment   : has both 'pin' and 'signal' fields.
      - LEDBehavior     : has both 'state' and 'meaning' fields.
      - ConnectorSpec   : has a 'pins' field (collection of pin assignments).
      - TechParameter   : has 'name' + 'value', AND no 'currency'/'amount'/'price'
                          (those signal a commercial/logistics item, not a tech param).
    Keys are matched case-insensitively.
    """
    if items_schema.get("type") != "object":
        return None
    props = items_schema.get("properties") or {}
    if not props:
        return None

    keys = {str(k).lower() for k in props.keys()}
    commercial_markers = {"currency", "amount", "price", "total"}

    if {"pin", "signal"} <= keys:
        return PinAssignment
    if {"state", "meaning"} <= keys:
        return LEDBehavior
    if "pins" in keys:
        return ConnectorSpec
    if {"name", "value"} <= keys and not (keys & commercial_markers):
        return TechParameter
    return None

logger = logging.getLogger(__name__)

class DynamicFieldDef(BaseModel):
    name: str
    type: str
    description: str
    items: Optional['DynamicFieldDef'] = None # For array items
    properties: Optional[Dict[str, 'DynamicFieldDef']] = None # For object properties

DynamicFieldDef.model_rebuild()

class DiscoveryResult(BaseModel):
    domain: str
    is_high_density: bool
    nested_skeleton: Dict[str, Any]
    confidence: float = 1.0

class DiscoveryAgent:
    """
    Agent responsible for scanning document content and synthesizing 
    the optimal Pydantic model for extraction dynamically.
    """

    def __init__(self, model_id: str = "gpt-4o"):
        self.client = VLMClient(config={"model": model_id})

    @staticmethod
    def _salvage_json(response: str) -> Optional[Dict[str, Any]]:
        """
        Pull a JSON object out of a model's raw text response.
        Used as a fallback when schema-constrained generation is unavailable
        (e.g. local fine-tuned extraction models). Returns None on failure
        rather than raising.
        """
        if not response:
            return None
        raw = response.strip()

        # 1. Strip code-fence markdown
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

        # 2. Locate the outermost JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            payload = match.group(0)
        else:
            first = raw.find("{")
            if first == -1:
                return None
            payload = raw[first:]

        # 3. Strict parse, then progressively forgiving repairs
        def _try(s: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None

        data = _try(payload)
        if data is not None:
            return data

        # Trim trailing junk past the last '}'
        last = payload.rfind("}")
        if last != -1:
            data = _try(payload[: last + 1])
            if data is not None:
                return data

        # Strip trailing commas before `]` or `}`
        cleaned = re.sub(r",\s*([\]}])", r"\1", payload)
        data = _try(cleaned)
        if data is not None:
            return data

        # Last resort: close braces (truncation recovery)
        for n in range(1, 11):
            candidate = cleaned + ("}" * n)
            data = _try(candidate)
            if data is not None:
                return data
        return None

    @staticmethod
    def _heuristic_domain(text: str) -> tuple:
        """
        Fast regex-based domain hint. Returns (domain, confidence_in_[0,1]).
        Confidence = best_score / total_score across all buckets; 0.0 if no signal.
        Intentionally inlined (not pulled from schema_engine) to avoid coupling.
        """
        t = (text or "").lower()
        buckets = {
            "Invoice": [
                r"\binvoice\b", r"\brechnung\b", r"\bbill to\b", r"\bship to\b",
                r"\bquantity\b", r"\bunit price\b", r"\bsubtotal\b",
                r"\bvat\b", r"\bgrand total\b", r"\bamount due\b",
            ],
            "Logistics": [
                r"\bquotation\b", r"\bquote no\b", r"\bport of (loading|destination)\b",
                r"\bbunker\b", r"\bcontainer\b", r"\bsurcharge\b",
                r"\bfreight\b", r"\btariff\b", r"\bbill of lading\b",
            ],
            "Industrial": [
                r"\bdatasheet\b", r"\bart\.?\s*no\b", r"\bsupply voltage\b",
                r"\bpin assignment\b|\bpinout\b", r"\bm12\b|\bm8\b",
                r"\bip\s*6[0-9]\b", r"\bconnector\b", r"\bled\b", r"\bdiagnostic\b",
            ],
            "Academic": [
                r"\babstract\b", r"\bmethodology\b", r"\breferences\b",
                r"\bconclusion\b", r"\bdoi:\b", r"\barxiv\b",
            ],
            "Legal": [
                r"\bagreement\b", r"\bhereby\b", r"\bclause\b",
                r"\bpursuant\b", r"\bterms and conditions\b", r"\bparty\b",
            ],
            "Medical_Record": [
                r"\bpatient\b", r"\bdiagnosis\b", r"\bprescription\b",
                r"\bmedical record\b", r"\bphysician\b", r"\bicd-?10\b",
            ],
        }
        scores = {k: sum(1 for p in v if re.search(p, t)) for k, v in buckets.items()}
        total = sum(scores.values())
        if total == 0:
            return ("General", 0.0)
        best = max(scores, key=scores.get)
        return (best, round(scores[best] / total, 2))

    @staticmethod
    def _default_skeleton_for_domain(domain: str) -> Dict[str, Any]:
        """
        Domain-typical default skeletons used as a Path C fallback when both
        LLM Discovery paths fail. Each skeleton lists the *common* fields for
        that domain — Pydantic synthesis later marks all of them Optional,
        so the extractor only fills what it actually finds in the document.

        These are general per-domain shapes, NOT tuned to any specific PDF.
        """
        skeletons: Dict[str, Dict[str, Any]] = {
            "Logistics": {
                "properties": {
                    "quote_identity": {
                        "type": "object", "description": "Quote header",
                        "properties": {
                            "quote_no":  {"type": "string", "description": "Quote number"},
                            "issuer":    {"type": "string", "description": "Issuing carrier"},
                            "recipient": {"type": "string", "description": "Customer / recipient"},
                            "quote_date": {"type": "string", "description": "Quote date"},
                            "expiration_date": {"type": "string", "description": "Quote expiration date"},
                        },
                    },
                    "route": {
                        "type": "object", "description": "Shipment route",
                        "properties": {
                            "origin":            {"type": "string", "description": "Origin"},
                            "destination":       {"type": "string", "description": "Destination"},
                            "port_of_loading":   {"type": "string", "description": "Port of loading"},
                            "port_of_discharge": {"type": "string", "description": "Port of discharge"},
                        },
                    },
                    "commodity": {"type": "string", "description": "Commodity / cargo description"},
                    "charges": {
                        "type": "array", "description": "Line-item charges",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "description": "What is being charged"},
                                "value":       {"type": "string", "description": "Numeric amount"},
                                "currency":    {"type": "string", "description": "Currency code"},
                            },
                        },
                    },
                },
            },
            "Invoice": {
                "properties": {
                    "invoice": {
                        "type": "object", "description": "Invoice container",
                        "properties": {
                            "identity": {
                                "type": "object", "description": "Invoice header",
                                "properties": {
                                    "number": {"type": "string", "description": "Invoice number"},
                                    "date":   {"type": "string", "description": "Invoice date"},
                                },
                            },
                            "from": {
                                "type": "object", "description": "Seller / issuer",
                                "properties": {
                                    "name":    {"type": "string", "description": "Seller name"},
                                    "address": {"type": "string", "description": "Seller address"},
                                },
                            },
                            "to": {
                                "type": "object", "description": "Buyer / recipient",
                                "properties": {
                                    "name":    {"type": "string", "description": "Buyer name"},
                                    "address": {"type": "string", "description": "Buyer address"},
                                },
                            },
                            "items": {
                                "type": "array", "description": "Invoice line items",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string", "description": "Item description"},
                                        "quantity":    {"type": "number", "description": "Quantity"},
                                        "unit_price":  {"type": "number", "description": "Unit price"},
                                        "total":       {"type": "number", "description": "Line total"},
                                    },
                                },
                            },
                            "grand_total": {"type": "number", "description": "Invoice grand total"},
                            "currency":    {"type": "string", "description": "Currency code"},
                        },
                    },
                },
            },
            "Industrial": {
                "properties": {
                    "identity": {
                        "type": "object", "description": "Product identity",
                        "properties": {
                            "product_name": {"type": "string", "description": "Product name"},
                            "art_no":       {"type": "string", "description": "Article number"},
                            "manufacturer": {"type": "string", "description": "Manufacturer"},
                        },
                    },
                    "parameters": {
                        "type": "array", "description": "Technical parameters",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":  {"type": "string", "description": "Parameter name"},
                                "value": {"type": "string", "description": "Parameter value"},
                                "unit":  {"type": "string", "description": "Unit of measurement"},
                            },
                        },
                    },
                    "connectors": {
                        "type": "array", "description": "Connectors and pinouts",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Connector type"},
                                "pins": {
                                    "type": "array", "description": "Pin assignments",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "pin":      {"type": "string", "description": "Pin number"},
                                            "signal":   {"type": "string", "description": "Signal name"},
                                            "function": {"type": "string", "description": "Signal function"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "diagnostics": {
                        "type": "array", "description": "LED / diagnostic states",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":    {"type": "string", "description": "LED / indicator name"},
                                "state":   {"type": "string", "description": "Color / behaviour"},
                                "meaning": {"type": "string", "description": "Diagnostic meaning"},
                            },
                        },
                    },
                    "standards": {
                        "type": "array", "description": "Standards and certifications",
                        "items": {"type": "string"},
                    },
                },
            },
            "Academic": {
                "properties": {
                    "title":      {"type": "string", "description": "Paper title"},
                    "abstract":   {"type": "string", "description": "Abstract"},
                    "authors":    {"type": "array", "description": "Author names", "items": {"type": "string"}},
                    "references": {"type": "array", "description": "References", "items": {"type": "string"}},
                },
            },
            "Legal": {
                "properties": {
                    "parties":       {"type": "array", "description": "Named parties", "items": {"type": "string"}},
                    "effective_date": {"type": "string", "description": "Effective date"},
                    "clauses": {
                        "type": "array", "description": "Clauses / terms",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string", "description": "Clause heading"},
                                "text":    {"type": "string", "description": "Clause text"},
                            },
                        },
                    },
                },
            },
            "Medical_Record": {
                "properties": {
                    "patient": {
                        "type": "object", "description": "Patient identification",
                        "properties": {
                            "name": {"type": "string", "description": "Patient name"},
                            "dob":  {"type": "string", "description": "Date of birth"},
                        },
                    },
                    "visit_date":    {"type": "string", "description": "Visit date"},
                    "diagnoses":     {"type": "array", "description": "Diagnoses", "items": {"type": "string"}},
                    "prescriptions": {"type": "array", "description": "Prescriptions", "items": {"type": "string"}},
                },
            },
        }
        # Aliases
        skeletons["Industrial_datasheet"] = skeletons["Industrial"]
        skeletons["Industrial_manual"]    = skeletons["Industrial"]
        skeletons["Industrial_product_pdf"] = skeletons["Industrial"]
        return skeletons.get(domain, {
            "properties": {
                "title":        {"type": "string", "description": "Document title"},
                "summary":      {"type": "string", "description": "High-level summary"},
                "key_entities": {"type": "array", "description": "Important entities",
                                 "items": {"type": "string"}},
            }
        })

    @staticmethod
    def _stratified_preview(doc_preview: str, max_chars: int = 8000) -> str:
        """
        Sample head + middle + tail of a long doc preview so rare modules
        (LED diagnostics, standards) that live in the middle are not missed.
        """
        if not doc_preview:
            return ""
        if len(doc_preview) <= max_chars:
            return doc_preview
        head_n = max_chars // 2
        tail_n = max_chars // 4
        mid_n = max_chars - head_n - tail_n
        mid_start = (len(doc_preview) - mid_n) // 2
        head = doc_preview[:head_n]
        middle = doc_preview[mid_start: mid_start + mid_n]
        tail = doc_preview[-tail_n:]
        return f"{head}\n\n[... mid-doc ...]\n\n{middle}\n\n[... tail ...]\n\n{tail}"

    def scout(self, doc_preview: str, graph_summary: str = "") -> DiscoveryResult:
        """
        Analyse a document preview and synthesise the optimal extraction schema.
        Uses the provider's schema-constrained generation path (`generate_structured`)
        so the heroic JSON salvage of v3 is no longer needed at this layer.
        """
        preview = self._stratified_preview(doc_preview, max_chars=8000)
        graph_snippet = (graph_summary or "")[:2000]

        # Heuristic pre-classification — fed to the LLM as a hint, not as a hard pin.
        # We score against both the preview and the graph summary so both content
        # and structural-keyword signals are considered.
        heur_text = (preview or "") + "\n" + (graph_snippet or "")
        heur_domain, heur_conf = self._heuristic_domain(heur_text)

        prompt = f"""You are a Senior Data Architect.
Analyse the document preview below and synthesise the optimal JSON schema for a
high-fidelity extraction. The schema MUST REFLECT WHAT IS ACTUALLY IN THIS
DOCUMENT — do not invent fields that aren't present.

### INSTRUCTIONS:
1. Identify the core 'domain'. Choose ONE of:
   Invoice, Logistics, Industrial, Industrial_datasheet, Industrial_product_pdf,
   Industrial_manual, Medical_Record, Academic, Legal, Corporate, General.
   Pick "Invoice" for bills/receipts with line items, totals, buyer+seller.
   Pick "Logistics" only for freight quotes, BoLs, shipping tariffs.
2. Set 'is_high_density' to true ONLY if you see dense technical specification
   tables or parameter-heavy content.
3. Build a 'nested_skeleton' that mirrors the document's own structure.
   - Use standard JSON Schema types ("string", "integer", "number", "boolean",
     "array", "object").
   - Every property MUST be a dict with `type` and `description`.
   - Use snake_case throughout.
   - DEEP nesting is appropriate when the document has hierarchical data (e.g.
     `electrical_specs.supply_voltage`); FLAT is better for simple docs.
4. ONLY include fields you can ground in the preview. Empty schemas are better
   than hallucinated ones.

### TYPED-BLOCK SHAPE HINTS (USE ONLY IF THE CORRESPONDING CONTENT APPEARS):
- Technical parameter tables → array of objects with {{name, value, unit}}.
- Connector pinouts          → array of objects with {{pin, signal, function}}.
- LED diagnostics            → array of objects with {{name, state, meaning}}.
- Invoice/quote line items   → array of objects with {{description, quantity, unit_price, total}}.

### FEW-SHOT EXAMPLE — HARDWARE DATASHEET:
{{
  "properties": {{
    "product_identity": {{
      "type": "object", "description": "Basic identification details",
      "properties": {{
        "art_no": {{"type": "string", "description": "The article number"}},
        "manufacturer": {{"type": "string", "description": "The company name"}}
      }}
    }},
    "technical_parameters": {{
      "type": "array", "description": "List of specifications",
      "items": {{
        "type": "object",
        "properties": {{
          "name":  {{"type": "string", "description": "Name of parameter"}},
          "value": {{"type": "string", "description": "The value"}},
          "unit":  {{"type": "string", "description": "The unit of measurement"}}
        }}
      }}
    }}
  }}
}}

### FEW-SHOT EXAMPLE — LOGISTICS QUOTE (no hardware fields):
{{
  "properties": {{
    "quote_identity": {{
      "type": "object", "description": "Quote header",
      "properties": {{
        "quote_no":   {{"type": "string", "description": "Quote number"}},
        "issuer":     {{"type": "string", "description": "Issuing carrier"}},
        "recipient":  {{"type": "string", "description": "Customer"}}
      }}
    }},
    "charges": {{
      "type": "array", "description": "Line-item charges",
      "items": {{
        "type": "object",
        "properties": {{
          "description": {{"type": "string", "description": "What is being charged"}},
          "value":       {{"type": "string", "description": "Numeric amount"}},
          "currency":    {{"type": "string", "description": "USD/EUR/..."}}
        }}
      }}
    }}
  }}
}}

### FEW-SHOT EXAMPLE — INVOICE / BILL (line items + parties + totals):
{{
  "properties": {{
    "invoice": {{
      "type": "object", "description": "Invoice container",
      "properties": {{
        "identity": {{
          "type": "object", "description": "Invoice header",
          "properties": {{
            "number": {{"type": "string", "description": "Invoice number"}},
            "date":   {{"type": "string", "description": "Invoice date"}}
          }}
        }},
        "from": {{
          "type": "object", "description": "Seller / issuer",
          "properties": {{
            "name":    {{"type": "string", "description": "Seller name"}},
            "address": {{"type": "string", "description": "Seller address"}}
          }}
        }},
        "to": {{
          "type": "object", "description": "Buyer / recipient",
          "properties": {{
            "name":    {{"type": "string", "description": "Buyer name"}},
            "address": {{"type": "string", "description": "Buyer address"}}
          }}
        }},
        "items": {{
          "type": "array", "description": "Invoice line items",
          "items": {{
            "type": "object",
            "properties": {{
              "description": {{"type": "string", "description": "Item description"}},
              "quantity":    {{"type": "number", "description": "Quantity"}},
              "unit_price":  {{"type": "number", "description": "Unit price"}},
              "total":       {{"type": "number", "description": "Line total"}}
            }}
          }}
        }},
        "grand_total": {{"type": "number", "description": "Invoice grand total"}}
      }}
    }}
  }}
}}

### HEURISTIC DOMAIN HINT (override only if clearly wrong):
Regex-based guess: {heur_domain} (confidence {heur_conf:.2f})

### DOCUMENT PREVIEW:
{preview}

### GRAPH STRUCTURE (sections & headings):
{graph_snippet if graph_snippet else '(none)'}

### OUTPUT:
Respond ONLY with a JSON object in exactly this shape:
{{
    "domain": "Detected_Domain",
    "is_high_density": true,
    "nested_skeleton": {{ "properties": {{ "...": {{"type": "string", "description": "..."}} }} }},
    "confidence": 0.9
}}
"""

        logger.info("DiscoveryAgent: Scouting document content...")

        # ── Path A: Schema-constrained generation (best for cloud LLMs) ──
        try:
            result = self.client.generate_structured(
                image=None,
                prompt=prompt,
                response_model=DiscoveryResult,
            )
            if (
                result
                and isinstance(result.nested_skeleton, dict)
                and "properties" in result.nested_skeleton
                and result.nested_skeleton["properties"]
            ):
                logger.info(
                    f"Discovery Success (structured): "
                    f"Domain=[{result.domain}] HighDensity=[{result.is_high_density}]"
                )
                return result
            logger.info("Structured discovery returned empty skeleton — trying raw-text path...")
        except Exception as e:
            logger.warning(f"Structured discovery failed: {e}. Trying raw-text path...")

        # ── Path B: Raw generate + JSON salvage (works for local extraction-tuned LLMs) ──
        try:
            response = self.client.generate(image=None, prompt=prompt) or ""
            data = self._salvage_json(response)
            if (
                data
                and isinstance(data.get("nested_skeleton"), dict)
                and "properties" in data["nested_skeleton"]
                and data["nested_skeleton"]["properties"]
            ):
                data.setdefault("confidence", 1.0)
                result = DiscoveryResult(**data)
                logger.info(
                    f"Discovery Success (raw): "
                    f"Domain=[{result.domain}] HighDensity=[{result.is_high_density}]"
                )
                return result
            logger.warning("Raw discovery produced invalid skeleton — trying heuristic fallback.")
        except Exception as e:
            logger.error(f"Raw scout failed: {e}")

        # ── Path C: Heuristic-domain default skeleton ──
        # Both LLM paths failed. If the regex heuristic confidently classified
        # the document, use a domain-typical default skeleton instead of the
        # 3-stub General fallback. Pydantic synthesis later marks every field
        # Optional, so the extractor only fills what it actually finds.
        if heur_domain and heur_domain != "General" and heur_conf >= 0.25:
            skeleton = self._default_skeleton_for_domain(heur_domain)
            logger.info(
                f"Discovery Fallback (heuristic): Domain=[{heur_domain}] "
                f"conf={heur_conf:.2f} — using domain-typical default skeleton."
            )
            return DiscoveryResult(
                domain=heur_domain,
                is_high_density=False,
                nested_skeleton=skeleton,
                confidence=heur_conf,
            )

        logger.warning("Heuristic confidence too low — using General fallback.")
        return DiscoveryResult(
            domain="General",
            is_high_density=False,
            nested_skeleton=self._default_skeleton_for_domain("General"),
            confidence=0.0,
        )

    def synthesize_model(self, result: DiscoveryResult) -> Type[BaseModel]:
        """Dynamically creates a Pydantic model from synthesized nested fields."""
        logger.info(f"DiscoveryAgent: Synthesizing model for {result.domain}...")
        
        # We can reuse the recursive builder from external schemas!
        def build_pydantic_model(schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            properties = schema.get("properties", {})
            fields = {}
            
            def normalize_prop(info: Any, name: str) -> Dict[str, Any]:
                if isinstance(info, str):
                    return {"type": info, "description": f"Extracted {name}"}
                if isinstance(info, list):
                    return {"type": "array", "description": f"Extracted {name}", "items": {"type": "string"}}
                if isinstance(info, dict):
                    return info
                return {"type": "string", "description": f"Extracted {name}"}

            for prop_name, prop_info_raw in properties.items():
                prop_info = normalize_prop(prop_info_raw, prop_name)
                
                prop_type = prop_info.get("type", "string")
                description = prop_info.get("description", "")
                
                # Recursive Case: Object
                if prop_type == "object" and "properties" in prop_info:
                    nested_name = "".join(x.capitalize() for x in prop_name.split("_")) + "Model"
                    python_type = build_pydantic_model(prop_info, nested_name)
                # Recursive Case: Array
                elif prop_type == "array" and "items" in prop_info:
                    items = prop_info.get("items", {})
                    # JSON Schema's `items` may be a single schema (the common
                    # case) or a list of schemas (the rare tuple form). When
                    # the LLM emits a tuple-form array, coerce to the first
                    # entry rather than crashing on `.get()`.
                    if isinstance(items, list):
                        items = items[0] if items else {}
                    if not isinstance(items, dict):
                        items = {}
                    if items.get("type") == "object":
                        typed_block = _match_typed_block(items)
                        if typed_block is not None:
                            logger.info(
                                f"      [Typed-Block] Injected {typed_block.__name__} for field '{prop_name}'"
                            )
                            python_type = List[typed_block]
                        else:
                            nested_name = "".join(x.capitalize() for x in prop_name.split("_")) + "Item"
                            inner_type = build_pydantic_model(items, nested_name)
                            python_type = List[inner_type]
                    else:
                        python_type = List[type_mapping.get(items.get("type", "string"), str)]
                # Base Case: Primitive
                else:
                    python_type = type_mapping.get(prop_type, str)

                import re
                def to_snake(name):
                    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

                snake_name = to_snake(prop_name)
                v_alias = AliasChoices(prop_name, snake_name)

                if prop_type == "array":
                    fields[prop_name] = (python_type, Field(default_factory=list, description=description, validation_alias=v_alias))
                elif prop_type == "object":
                    fields[prop_name] = (Optional[python_type], Field(None, description=description, validation_alias=v_alias))
                else:
                    fields[prop_name] = (Optional[python_type], Field(None, description=description, validation_alias=v_alias))

            return create_model(model_name, __base__=BaseModel, **fields)

        root_name = f"Dynamic{result.domain.replace('_', '').replace(' ', '')}Schema"
        root_name = re.sub(r'[^a-zA-Z0-9]', '', root_name)
        if not root_name.isidentifier() or root_name == "DynamicSchema":
            root_name = "DynamicDocumentSchema"
            
        dyn_model = build_pydantic_model(result.nested_skeleton, root_name)
        
        # Add universal metadata fields
        extra_fields = {}
        extra_fields["reasoning_thoughts"] = (Optional[str], Field(None, description="Step-by-step reasoning for the extraction"))
        extra_fields["page_references"] = (List[int], Field(default_factory=list, description="Page numbers (integers) supporting the extraction. Use the <!-- page:N --> markers in the source markdown."))
        extra_fields["source_evidence"] = (List[SourceEvidence], Field(default_factory=list, description="3–6 short source snippets (≤200 chars each) with page numbers grounding the top extractions."))
        extra_fields["confidence_score"] = (float, Field(1.0))

        dyn_model = create_model(f"{root_name}WithMeta", __base__=dyn_model, **extra_fields)
        return dyn_model
    def synthesize_from_external_schema(self, schema_path: str) -> Type[BaseModel]:
        """Loads a JSON schema from disk and synthesizes a Pydantic model."""
        logger.info(f"DiscoveryAgent: Loading external schema from {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_dict = json.load(f)
            
        def build_pydantic_model(schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            fields = {}
            
            for prop_name, prop_info in properties.items():
                prop_type = prop_info.get("type")
                description = prop_info.get("description", "")
                
                # Recursive Case: Object
                if prop_type == "object":
                    nested_name = "".join(x.capitalize() for x in prop_name.split("_")) + "Model"
                    python_type = build_pydantic_model(prop_info, nested_name)
                # Recursive Case: Array
                elif prop_type == "array":
                    items = prop_info.get("items", {})
                    if items.get("type") == "object":
                        nested_name = "".join(x.capitalize() for x in prop_name.split("_")) + "Item"
                        inner_type = build_pydantic_model(items, nested_name)
                        python_type = List[inner_type]
                    else:
                        python_type = List[type_mapping.get(items.get("type", "string"), str)]
                # Base Case: Primitive
                else:
                    python_type = type_mapping.get(prop_type, str)
                
                import re
                def to_snake(name):
                    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
                
                snake_name = to_snake(prop_name)
                # Use AliasChoices to support both original and snake_case
                v_alias = AliasChoices(prop_name, snake_name)
                
                # RELAXATION: Treat everything as Optional for LLM robustness
                if prop_type == "array":
                    fields[prop_name] = (python_type, Field(default_factory=list, description=description, validation_alias=v_alias))
                elif prop_type == "object":
                    fields[prop_name] = (Optional[python_type], Field(None, description=description, validation_alias=v_alias))
                else:
                    fields[prop_name] = (Optional[python_type], Field(None, description=description, validation_alias=v_alias))
            
            return create_model(model_name, __base__=BaseModel, **fields)

        # Build the master model
        root_name = schema_dict.get("title", "ExternalSchema").replace(" ", "")
        master_model = build_pydantic_model(schema_dict, root_name)
        
        # Inject standard metadata fields if not present
        extra_fields = {}
        if "reasoning_thoughts" not in master_model.model_fields:
            extra_fields["reasoning_thoughts"] = (Optional[str], Field(None, description="Step-by-step reasoning"))
        if "page_references" not in master_model.model_fields:
            extra_fields["page_references"] = (List[int], Field(default_factory=list))
        if "confidence_score" not in master_model.model_fields:
            extra_fields["confidence_score"] = (float, 1.0)
            
        if extra_fields:
            master_model = create_model(f"{root_name}WithMeta", __base__=master_model, **extra_fields)
            
        return master_model
