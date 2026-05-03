"""
Schema Engine: Adaptive Librarian Audit
=======================================
The 'Scout' of the Librarian Agent. 
Performs a high-level scan of the DocumentGraph to determine which
technical modules are required for the current document.
"""

import logging
import re
from typing import List, Type, Dict, Any, Optional
from pydantic import BaseModel
from core.schemas import DocumentGraph, HierarchicalNode
from common.vlm_client import VLMClient
from extractor.schema_definitions import LibrarianUniversalHardware
from extractor.schema_registry import (
    get_schema_for_domain,
    get_active_modules_for_domain,
    get_schema_family,
    get_schema_model,
)

logger = logging.getLogger(__name__)

class SchemaAuditor:
    """
    Audits the document to propose the most efficient extraction schema.
    """
    
    def __init__(self, model_id: str = "gpt-4o"):
        self.client = VLMClient(config={"model": model_id})

    @staticmethod
    def _heuristic_domain_from_text(text: str) -> Dict[str, Any]:
        """Fast deterministic routing for obvious document families."""
        t = (text or "").lower()

        buckets = {
            "Corporate": [
                r"\binvoice\b",
                r"\bquotation\b",
                r"\bquote\b",
                r"\bport of loading\b",
                r"\bport of destination\b",
                r"\bsurcharge\b",
                r"\bcurrency\b",
                r"\busd\b|\beur\b|\bgbp\b",
                r"\bper container\b",
                r"\btotal\b",
            ],
            "Industrial": [
                r"\bvoltage\b",
                r"\bcurrent\b",
                r"\bconnector\b",
                r"\bpin\b",
                r"\bled\b",
                r"\bm12\b|\bm8\b",
                r"\bdatasheet\b",
                r"\bpart number\b|\bart\.? no\b",
            ],
            "Academic": [
                r"\babstract\b",
                r"\bmethodology\b",
                r"\breferences\b",
                r"\bconclusion\b",
                r"\bcitation\b",
            ],
            "Legal": [
                r"\bagreement\b",
                r"\bhereby\b",
                r"\bclause\b",
                r"\bparty\b",
                r"\bpursuant\b",
                r"\bterms and conditions\b",
            ],
        }

        scores: Dict[str, int] = {k: 0 for k in buckets.keys()}
        for domain, patterns in buckets.items():
            for pattern in patterns:
                if re.search(pattern, t):
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get) if scores else "General"
        best_score = scores.get(best_domain, 0)
        total_signal = sum(scores.values())

        if total_signal == 0:
            return {
                "domain": "General",
                "confidence": 0.0,
                "source": "heuristic",
                "signals": scores,
            }

        confidence = best_score / max(total_signal, 1)
        return {
            "domain": best_domain,
            "confidence": round(confidence, 4),
            "source": "heuristic",
            "signals": scores,
        }

    @staticmethod
    def _heuristic_industrial_subtype(text: str) -> str:
        """Determine if an Industrial document is a product PDF, technical datasheet, or technical manual.

        Datasheet signals: dense parameter tables, short product-spec focus, Art.No header.
        Manual signals: TOC/chapter structure, installation steps, safety section.
        Product PDF signals: catalog-like article sheets with commercial metadata and cable summaries.
        Defaults to 'technical_datasheet' when ambiguous.
        """
        t = (text or "").lower()

        manual_signals = [
            r"\btable of contents\b",
            r"\bchapter\s+\d",
            r"\binstallation\b",
            r"\bsafety\s+instructions?\b",
            r"\bconfiguration\b",
            r"\bfirmware\s+update\b",
            r"\btroubleshoot",
            r"\bwiring\s+diagram\b",
            r"\bmounting\s+instructions?\b",
            r"\bprocedure\b",
        ]
        product_pdf_signals = [
            r"\bproduct-pdf\b",
            r"\bfor article\b",
            r"\bart\.?\s*no\b",
            r"\bcable length\b",
            r"\bpackaging unit\b",
            r"\bgtin\b",
            r"\beclass[-\s]\d",
            r"\betim[-\s]\d",
            r"\bcustoms tariff number\b",
            r"\bno\.? of poles\b",
            r"\bfamily construction form\b",
        ]
        datasheet_signals = [
            r"\bdatasheet\b",
            r"\bdata\s+sheet\b",
            r"\btechnical\s+specifications?\b",
            r"\bspecification\s+sheet\b",
            r"\bart\.?\s*no\.?\b|\bart\.?\s*nr\.?\b",
            r"\boperating\s+voltage\b",
            r"\bsupply\s+voltage\b",
            r"\bprotection\s+class\b|\bip\s*6[0-9]\b",
            r"\bm12\b|\bm8\b",
            r"\bpin\s+assignment\b|\bpinout\b",
        ]

        manual_score = sum(1 for p in manual_signals if re.search(p, t))
        product_pdf_score = sum(1 for p in product_pdf_signals if re.search(p, t))
        datasheet_score = sum(1 for p in datasheet_signals if re.search(p, t))

        if product_pdf_score >= max(datasheet_score, manual_score) and product_pdf_score >= 2:
            return "product_pdf"
        return "technical_manual" if manual_score > datasheet_score else "technical_datasheet"

    @staticmethod
    def build_runtime_schema_draft(markdown: str, title: str = "Runtime Content Schema") -> Dict[str, Any]:
        """Generate a lightweight JSON schema draft from observed document cues."""
        t = (markdown or "").lower()

        def _has_any(patterns: List[str]) -> bool:
            return any(re.search(p, t) for p in patterns)

        properties: Dict[str, Any] = {
            "document_title": {"type": ["string", "null"]},
            "summary": {"type": ["string", "null"]},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                    },
                },
            },
            "tables_markdown": {"type": "array", "items": {"type": "string"}},
            "page_references": {"type": "array", "items": {"type": "integer"}},
        }

        # Commercial cues
        if _has_any([
            r"\binvoice\b",
            r"\bquotation\b",
            r"\bsurcharge\b",
            r"\bper\s+container\b",
            r"\bcurrency\b",
            r"\btotal\b",
        ]):
            properties.update({
                "invoice_number": {"type": ["string", "null"]},
                "quotation_number": {"type": ["string", "null"]},
                "supplier": {"type": ["string", "null"]},
                "recipient": {"type": ["string", "null"]},
                "currency": {"type": ["string", "null"]},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": ["string", "null"]},
                            "amount": {"type": ["string", "null"]},
                        },
                    },
                },
            })

        # Technical cues
        if _has_any([
            r"\bvoltage\b",
            r"\bconnector\b",
            r"\bpins?\b",
            r"\bm12\b",
            r"\bdatasheet\b",
            r"\bled\b",
        ]):
            properties.update({
                "product_name": {"type": ["string", "null"]},
                "manufacturer": {"type": ["string", "null"]},
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": ["string", "null"]},
                        },
                    },
                },
            })

        # Product PDF cues
        if _has_any([
            r"\bproduct-pdf\b",
            r"\bfor article\b",
            r"\bart\.?\s*no\b",
            r"\bcable length\b",
            r"\bpackaging unit\b",
            r"\bgtin\b",
            r"\beclass[-\s]\d",
            r"\betim[-\s]\d",
        ]):
            properties.update({
                "product_name": {"type": ["string", "null"]},
                "art_no": {"type": ["string", "null"]},
                "manufacturer": {"type": ["string", "null"]},
                "connector_type": {"type": ["string", "null"]},
                "pole_count": {"type": ["string", "null"]},
                "cable_length": {"type": ["string", "null"]},
                "material": {"type": ["string", "null"]},
                "protection_rating": {"type": ["string", "null"]},
                "electrical_specs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": ["string", "null"]},
                        },
                    },
                },
                "standards": {"type": "array", "items": {"type": "string"}},
                "commercial_data": {
                    "type": "object",
                    "properties": {
                        "gtin": {"type": ["string", "null"]},
                        "customs_tariff_number": {"type": ["string", "null"]},
                        "packaging_unit": {"type": ["string", "null"]},
                        "eclass_codes": {"type": "array", "items": {"type": "string"}},
                        "etim_codes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            })

        return {
            "title": title,
            "type": "object",
            "properties": properties,
            "required": [],
        }
        
    def audit_document(self, graph: DocumentGraph) -> Dict[str, Any]:
        """
        Scans the first 2-3 pages (or index nodes) of the HKG
        to detect document domain and find technical modules.
        """
        # 1. Collect candidate nodes (Title and first content nodes)
        candidate_nodes = []
        for node in graph.nodes[:5]:
            candidate_nodes.append(node.content)
            
        context_prompt = "\n---\n".join(candidate_nodes)

        heuristic = self._heuristic_domain_from_text(context_prompt)
        if heuristic["confidence"] >= 0.75:
            domain = heuristic["domain"]
            result = {
                "domain": domain,
                "found_electrical": domain == "Industrial",
                "found_mechanical": domain == "Industrial",
                "found_connectors": domain == "Industrial",
                "found_diagnostics": domain == "Industrial",
                "routing_confidence": heuristic["confidence"],
                "routing_source": "heuristic",
                "routing_signals": heuristic.get("signals", {}),
            }
            if domain == "Industrial":
                result["industrial_subtype"] = self._heuristic_industrial_subtype(context_prompt)
            return result
        
        audit_prompt = f"""
You are a Professional Document Librarian. 
Below is a 'preview' of a new document. 
Identify the DOCUMENT DOMAIN and check if specific data modules are present.

### PREVIEW:
{context_prompt}

### DOMAINS:
- Industrial: Technical manuals, data sheets, PLC guides.
- Academic: Research papers, thesis, scientific reports.
- Legal: Contracts, patents, legal notices.
- Corporate: Invoices, business reports, case studies.
- General: Miscellaneous text, books, letters.

### TECHNICAL MODULES (Only for Industrial):
1. ELECTRICAL_SPECS: Operating voltage, current, supply.
2. MECHANICAL_SPECS: Dimensions, weight, material.
3. CONNECTOR_PINOUTS: Pin assignments for M12, M8, etc.
4. DIAGNOSTICS: LED states or error codes.

### OUTPUT:
Respond ONLY with a JSON object like this:
{{
  "domain": "Industrial|Academic|Legal|Corporate|General",
  "found_electrical": true/false,
  "found_mechanical": true/false,
  "found_connectors": true/false,
  "found_diagnostics": true/false
}}
"""
        logger.info("Librarian Scout: Auditing document structure...")
        
        try:
            response_str = self.client.generate(image=None, prompt=audit_prompt)
            
            import json
            import re
            match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if match:
                results = json.loads(match.group(0))
                results["routing_confidence"] = float(results.get("routing_confidence", 0.6))
                results["routing_source"] = "vlm"
                results["routing_signals"] = heuristic.get("signals", {})
                if results.get("domain") == "Industrial":
                    results["industrial_subtype"] = self._heuristic_industrial_subtype(context_prompt)
                logger.info(f"Audit Complete: Domain=[{results.get('domain')}]")
                return results
        except Exception as e:
            logger.error(f"Audit Failed: {e}. Defaulting to Generic Industrial.")
            
        return {
            "domain": "Industrial",
            "found_electrical": True,
            "found_mechanical": True,
            "found_connectors": True,
            "found_diagnostics": True,
            "routing_confidence": 0.0,
            "routing_source": "fallback",
            "routing_signals": heuristic.get("signals", {}),
            "industrial_subtype": self._heuristic_industrial_subtype(context_prompt),
        }

    def get_discovery_schema(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a schema description for the ExtractorAgent based on domain audit.
        For Industrial documents, routes to product_pdf_v1, technical_datasheet_v1,
        or hardware_v1 based on the industrial_subtype detected during auditing.
        """
        domain = audit_results.get("domain", "General")
        industrial_subtype = audit_results.get("industrial_subtype", "technical_datasheet")

        # Resolve effective domain for Industrial sub-types
        effective_domain = domain
        if domain == "Industrial":
            if industrial_subtype == "product_pdf":
                effective_domain = "Industrial_product_pdf"
            elif industrial_subtype == "technical_datasheet":
                effective_domain = "Industrial_datasheet"
            else:
                effective_domain = "Industrial_manual"
            logger.info(f"Industrial sub-type detected: {industrial_subtype} → routing to {effective_domain}")

        active_modules = get_active_modules_for_domain(effective_domain)

        # For Industrial: filter modules to what audit confirmed is present
        if domain == "Industrial":
            found_modules = []
            if audit_results.get("found_electrical"): found_modules.append("Electrical Parameters")
            if audit_results.get("found_mechanical"): found_modules.append("Mechanical Parameters")
            if audit_results.get("found_connectors"): found_modules.append("Connector Pin Assignments")
            if audit_results.get("found_diagnostics"): found_modules.append("LED Diagnostics")
            if found_modules:
                active_modules = found_modules
            if industrial_subtype == "product_pdf":
                active_modules = [
                    "Product Identity",
                    "Connector Summary",
                    "Electrical Parameters",
                    "Standards & Certifications",
                    "Commercial Data",
                ]
            if industrial_subtype == "technical_datasheet" and "Standards & Certifications" not in active_modules:
                active_modules.append("Standards & Certifications")

        schema_class = get_schema_for_domain(effective_domain)
        schema_family = get_schema_family(effective_domain)
        
        return {
            "title": f"Librarian {effective_domain} Discovery",
            "domain": effective_domain,
            "active_modules": active_modules,
            "recommended_schema_family": schema_family,
            "base_model": schema_class.__name__,
            "routing_confidence": audit_results.get("routing_confidence", 0.0),
            "routing_source": audit_results.get("routing_source", "unknown"),
            "routing_signals": audit_results.get("routing_signals", {}),
        }

    def build_target_schema_contract(
        self,
        discovery: Dict[str, Any],
        explicit_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the final extraction contract.
        - If explicit_schema is provided, use it directly.
        - Otherwise, resolve from domain discovery to schema family/model.
        """
        if explicit_schema:
            return {
                "mode": "explicit",
                "schema_family": "explicit",
                "schema_title": explicit_schema.get("title", "Explicit Runtime Schema"),
                "schema_json": explicit_schema,
                "model_name": None,
            }

        domain = discovery.get("domain", "General") if discovery else "General"
        schema_family = discovery.get("recommended_schema_family", get_schema_family(domain)) if discovery else get_schema_family(domain)
        schema_model = get_schema_model(schema_family)
        schema_json = schema_model.model_json_schema()

        return {
            "mode": "domain",
            "domain": domain,
            "schema_family": schema_family,
            "schema_title": schema_json.get("title", schema_model.__name__),
            "schema_json": schema_json,
            "model_name": schema_model.__name__,
        }
