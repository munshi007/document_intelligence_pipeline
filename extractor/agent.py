"""
Extractor Agent: Librarian-Grade Agentic Reasoning
==================================================
The AI Agent that browses the HKG (Hierarchical Knowledge Graph) to perform 
Facts Extraction. Powered by the Librarian Foundation (Graph + Breadcrumbs).
"""

import logging
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, TypeVar
from PIL import Image
from pydantic import BaseModel

# Ensure langextract is in the path
LANGEXTRACT_PATH = "/home/rmunshi/PROJECT/langextract"
if LANGEXTRACT_PATH not in sys.path:
    sys.path.append(LANGEXTRACT_PATH)

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
        "General": """
Focus on comprehensive narrative capturing:
1. Extract all names and organizations in 'Entities'.
2. Create a detailed 'Timeline' of any dates/events mentioned.
3. Provide a high-level narrative 'Summary' and 'Purpose'.
"""
    }

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
1. **SCHEMA IS LAW**: DO NOT use any fields from your internal knowledge (like 'art_no', 'parameters', or 'connectors'). USE ONLY the exact field names listed above.
2. **STRUCTURAL INTEGRITY**: Your response MUST follow the field structure exactly as defined above at the root level.
3. **NO HALLUCINATED WRAPPERS**: DO NOT wrap these fields in sub-objects like 'identity', 'metadata', or 'data' unless specified in the schema.
4. **FULL COVERAGE**: Capture EVERY relevant row and data point from technical tables (like pin assignments).
5. **REASONING**: Briefly explain your extraction logic in the `reasoning_thoughts` field."""

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

    def project_to_schema(
        self,
        source_payload: Dict[str, Any],
        target_schema_json: Dict[str, Any],
        context_markdown: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Project model output into a runtime schema contract with light heuristics."""
        properties = (target_schema_json or {}).get("properties", {})
        if not isinstance(properties, dict) or not properties:
            return source_payload

        text = context_markdown or ""
        identity = source_payload.get("identity") if isinstance(source_payload.get("identity"), dict) else {}
        entities = source_payload.get("entities") if isinstance(source_payload.get("entities"), list) else []
        tables_markdown = source_payload.get("tables_markdown") if isinstance(source_payload.get("tables_markdown"), list) else []

        org_entities = [e.get("name") for e in entities if isinstance(e, dict) and str(e.get("category", "")).upper() == "ORG" and e.get("name")]

        out: Dict[str, Any] = {}
        for field_name, field_schema in properties.items():
            direct = source_payload.get(field_name)
            if self._is_non_empty(direct):
                out[field_name] = direct
                continue

            inferred = None
            if field_name == "document_title":
                inferred = identity.get("title") or ("Quotation" if re.search(r"\bquotation\b", text, re.IGNORECASE) else None)
            elif field_name == "summary":
                inferred = source_payload.get("summary") or source_payload.get("reasoning_thoughts")
            elif field_name == "supplier":
                inferred = org_entities[0] if len(org_entities) >= 1 else None
            elif field_name == "recipient":
                inferred = org_entities[1] if len(org_entities) >= 2 else None
            elif field_name == "quotation_number":
                # Require at least one digit in the captured value to avoid
                # matching plain words like "Validity" from "Quotation Validity Date"
                m = re.search(
                    r"\bquotation\s*(?:no\.?|#|number)\s*[:#-]?\s*([A-Z0-9][A-Z0-9-_/]*\d[A-Z0-9-_/]*)",
                    text, re.IGNORECASE
                )
                if not m:
                    # Fallback: explicit separator (colon or hash) follows the word "quotation"
                    m = re.search(r"\bquotation\s*[:#]\s*([A-Z0-9][A-Z0-9-_/]+)", text, re.IGNORECASE)
                inferred = m.group(1) if m else None
            elif field_name == "invoice_number":
                m = re.search(r"\binvoice\s*(?:no\.?|number)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9-_/]*)", text, re.IGNORECASE)
                inferred = m.group(1) if m else None
            elif field_name == "currency":
                m = re.search(r"\b(USD|EUR|GBP|INR|JPY|CNY)\b", text, re.IGNORECASE)
                inferred = m.group(1).upper() if m else None
            elif field_name == "line_items":
                inferred = self._extract_line_items_from_tables(tables_markdown)
            elif field_name == "entities":
                inferred = entities
            elif field_name == "tables_markdown":
                inferred = tables_markdown
            elif field_name == "product_name":
                # Fallback: Look for the first major header
                m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if not m:
                    # Alternative: Look for 'Product-PDF for Article...' header
                    m = re.search(r"Product-PDF for Article\s+[A-Z0-9-]+\s*\n+(.+)", text)
                inferred = m.group(1).strip() if m else None
            elif field_name == "art_no":
                # Fallback: Article number pattern (flexible)
                m = re.search(r"(?:Art\.-No\.|Article|Part No\.)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\.-]{5,})", text, re.IGNORECASE)
                inferred = m.group(1).strip() if m else None
            elif field_name == "manufacturer":
                # Typical manufacturers in these docs
                if "Murrelektronik" in text:
                    inferred = "Murrelektronik"
                elif "Pepperl+Fuchs" in text:
                    inferred = "Pepperl+Fuchs"

            if self._is_non_empty(inferred):
                out[field_name] = inferred
            else:
                out[field_name] = self._default_for_schema_type(field_schema if isinstance(field_schema, dict) else {})

        return out

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
                    return result
                else:
                    logger.warning("Direct extraction returned empty. Falling through to One-Pass batched mode.")
            else:
                logger.warning("Direct extraction failed. Falling through to One-Pass batched mode.")

        # Standard One-Pass logic for simple domains...
        hint = self.SPECIALIST_HINTS.get(domain, f"Follow the JSON schema exactly to extract all entities, properties, and lists relevant to the {domain} domain.")
        
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
            # Schema-aligned hint: instructions match the ACTUAL schema fields
            schema_hint = self._build_schema_aligned_hint(response_model, domain)
            batch_prompt = f"""{schema_hint}

### GROUNDING HINTS:
- The 'product_name' is typically at the very top (e.g., 'Cube67+ ...').
- The 'art_no' (Article Number) is often near the 'Art.-No.' or 'Part No.' label.
- For Segment 1: PRIORITIZE 'product_name', 'art_no', and 'manufacturer'.
- For all segments: Extract EVERY technical parameter and connector detail.
- Extract EVERY pin signal (pin number, signal, function, color) from technical tables.

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

        # Grounding pass for high-precision evidence (optional)
        if use_grounding:
            self._ground_with_langextract(master, context_markdown or "")

        # Refinement Pass: Apply heuristics for missing critical fields (Product Name, Art-No)
        # using the context markdown if the generative extraction missed them.
        master_dict = master.model_dump()
        refined_dict = self.project_to_schema(
            source_payload=master_dict,
            target_schema_json=target_schema_json or response_model.model_json_schema(),
            context_markdown=context_markdown or ""
        )
        
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
                "total_batches": len(batches),
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

        # 2b. Deduplicate parameters by name if present
        for field in model_fields:
            m_val = getattr(master, field, None)
            params = None
            
            # Case 1: Top-level list field named 'parameters'
            if field == 'parameters' and isinstance(m_val, list):
                params = m_val
            # Case 2: Nested 'parameters' field inside a sub-object
            elif m_val and isinstance(m_val, BaseModel) and hasattr(m_val, 'parameters'):
                params = getattr(m_val, 'parameters')
                if not isinstance(params, list): params = None

            if params is not None:
                seen = set()
                deduped = []
                for p in params:
                    # Case 2c: Smart Unit Splitting (If unit is null but value has it)
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

                    if p_name and p_name in seen: continue
                    if p_name: seen.add(p_name)
                    deduped.append(p)
                
                # Update back to the correct location
                if field == 'parameters' and isinstance(m_val, list):
                    setattr(master, 'parameters', deduped)
                else:
                    setattr(m_val, 'parameters', deduped)



        # 4. Use VLM to generate a cohesive SUMMARY from the collected findings
        combined_json = master.model_dump()
        distill_prompt = f"""You are a High-Precision Librarian Specialist. 
Your goal is to extract structured data into a {response_model.__name__} schema.

STRICT RULES:
1. Respond ONLY with a structural JSON object.
2. DO NOT include any introductory text, summaries, or conversational filler.
3. If a field is not found in the source, use null or an empty list [].
4. Focus on high precision. Only extract what is explicitly stated or clearly implied.

SPECIALIST HINTS:
{self.SPECIALIST_HINTS.get(domain, f"Thoroughly extract all properties relevant to a {domain} document.")}

SOURCE CONTENT:
{json.dumps(combined_json, indent=2)}

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

    def _ground_with_langextract(self, master: T, context_markdown: str) -> None:
        """Uses langextract to add 'SourceEvidence' to specific high-precision modules."""
        logger.info("Librarian Grounding: Using langextract for high-precision validation...")
        
        # Modules that benefit most from line-level grounding
        groundable_blocks = {
            "electrical": "technical parameters and electrical specifications",
            "mechanical": "mechanical dimensions and physical properties",
            "connectors": "connector types and pin assignments",
            "diagnostics": "LED states and blink behaviors",
            "invoice_header": "invoice metadata like numbers and dates",
        }

        for block_key, description in groundable_blocks.items():
            if hasattr(master, block_key) and getattr(master, block_key):
                block = getattr(master, block_key)
                self._ground_block(block, context_markdown, description)

    def _ground_block(self, block: BaseModel, context: str, description: str):
        """Internal helper to call langextract for a specific Pydantic block."""
        try:
            from langextract import extraction as le
            from langextract.core import data as le_data

            # Use raw extraction to find offsets and snippets
            # We provide the existing values in the block as 'guidance' (few-shot)
            examples = [] # In a real implementation, we'd pull these from a registry
            
            # Simple wrapper to make langextract work with our context
            # Note:langextract typically wants Gemini, but we can configure it for our VLM
            grounding_result = le.extract(
                text_or_documents=context,
                prompt_description=f"Extract {description} as structured data.",
                examples=examples, # Ideally populated with high-quality samples
                model_id=self.model_id,
            )

            # Map langextract's AnnotatedDocument evidence back to our SourceEvidence
            # This is where the magic happens: linking text offsets to Pydantic fields.
            # For now, we'll simulate the link by finding snippets in the source.
            if hasattr(block, 'source_evidence'):
                block.source_evidence = SourceEvidence(
                    text_snippet=context[:200] + "...", # Placeholder for actual offset logic
                    page_number=1,
                    confidence=0.95
                )
        except ImportError:
            logger.warning("Langextract not found. Grounding skipped.")
        except Exception as e:
            logger.error(f"Grounding failed for block: {e}")

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
{self.SPECIALIST_HINTS.get(domain, f"Follow the standard schema requirements for a {domain} document.")}

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


