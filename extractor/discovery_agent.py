import logging
import json
import re
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, create_model, Field, AliasChoices
from common.vlm_client import VLMClient
from extractor.schema_definitions import (
    IdentificationBlock,
    TechParameter,
    ConnectorSpec,
    LEDBehavior,
    InvoiceHeaderBlock,
    InvoiceLinesBlock,
    TotalsBlock,
    GeneralInfoBlock,
    SourceEvidence,
)

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

    def scout(self, doc_preview: str, graph_summary: str = "") -> DiscoveryResult:
        """Analyzes a document preview to identify required data modules."""
        prompt = f"""
You are a Senior Data Architect. 
Analyze the document preview and knowledge graph below and synthesize the optimal JSON schema required for a high-fidelity extraction.

### INSTRUCTIONS:
1. Identify the core 'domain' (e.g., Medical_Record, Technical_Datasheet, Invoice, Industrial).
2. Set 'is_high_density' to `true` if you see dense technical tables or specifications.
3. Define a 'nested_skeleton' dictionary mapping the data hierarchy.
   - Use standard JSON Schema types ("string", "integer", "number", "boolean", "array", "object").
   - CRITICAL: For Technical/Industrial docs, use DEEP nesting to group structural data (e.g., `electrical_specs`).
   - For technical parameters, ALWAYS use an array of objects with `name`, `value`, and `unit`.
   - Every property MUST be a dictionary with "type" and "description".
   - Use snake_case for names.
4. If the document is a standard format (Invoice, Receipt), keep the structure FLAT and focused on 10-15 core fields.

### FEW-SHOT EXAMPLE OF EXPECTED `nested_skeleton` FORMAT:
"nested_skeleton": {{
    "properties": {{
        "product_identity": {{
            "type": "object",
            "description": "Basic identification details",
            "properties": {{
                "art_no": {{"type": "string", "description": "The article number"}},
                "manufacturer": {{"type": "string", "description": "The company name"}}
            }}
        }},
        "technical_parameters": {{
            "type": "array",
            "description": "List of specifications",
            "items": {{
                "type": "object",
                "properties": {{
                    "name": {{"type": "string", "description": "Name of parameter"}},
                    "value": {{"type": "string", "description": "The numeric value"}},
                    "unit": {{"type": "string", "description": "The unit of measurement"}}
                }}
            }}
        }}
    }}
}}

### PREVIEW:
{doc_preview[:5000]}

### OUTPUT:
Respond ONLY with the JSON object:
{{
    "domain": "Detected_Domain",
    "is_high_density": true/false,
    "nested_skeleton": {{ "properties": {{ ... }} }},
    "confidence": 0.9
}}
"""
        logger.info("DiscoveryAgent: Scouting document content...")
        response = ""
        try:
            response = self.client.generate(image=None, prompt=prompt)
            raw_resp = response.strip()
            
            # 1. Strip markdown
            clean_resp = raw_resp
            if "```json" in clean_resp:
                clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_resp:
                clean_resp = clean_resp.split("```")[1].split("```")[0].strip()

            # 2. Extract JSON block (greedy start, greedy end)
            match = re.search(r'(\{.*\})', clean_resp, re.DOTALL)
            if not match:
                # If no brackets, maybe it's truncated? Try to find first {
                first_brace = clean_resp.find('{')
                if first_brace != -1:
                    json_payload = clean_resp[first_brace:]
                else:
                    logger.error("Scout failed: No brackets found in response.")
                    raise ValueError("No JSON found")
            else:
                json_payload = match.group(1)
            
            # 3. Robust Truncated JSON Salvager
            def find_valid_json(s):
                s = s.strip()
                if not s: return None, None
                
                # Try parsing as is
                try:
                    return json.loads(s), s
                except json.JSONDecodeError:
                    pass
                
                # Try trimming trailing junk (comments, text)
                last_brace = s.rfind('}')
                if last_brace != -1:
                    try:
                        trimmed = s[:last_brace+1]
                        return json.loads(trimmed), trimmed
                    except:
                        pass
                
                # Last Resort: Truncation Recovery (Brute force close braces)
                # If it's cut off mid-string or mid-object
                temp_s = s
                for _ in range(10): # Try closing up to 10 braces
                    temp_s += "}"
                    try:
                        return json.loads(temp_s), temp_s
                    except:
                        continue
                
                raise ValueError("Could not salvage JSON")

            try:
                data, _ = find_valid_json(json_payload)
            except Exception as e:
                # regex-based emergency clean for trailing commas
                json_payload = re.sub(r",\s*([\]}])", r"\1", json_payload)
                try:
                    data, _ = find_valid_json(json_payload)
                except:
                    logger.error(f"Raw Scout Response:\n{response}")
                    raise e

            # 4. Normalize and Validate
            if "confidence" not in data:
                data["confidence"] = 1.0
            
            if "nested_skeleton" not in data or "properties" not in data.get("nested_skeleton", {}):
                logger.warning("Discovery failed to produce valid nested_skeleton. Using fallback.")
                raise ValueError("Invalid skeleton")

            logger.info(f"Discovery Success: Domain=[{data.get('domain')}] HighDensity=[{data.get('is_high_density')}]")
            return DiscoveryResult(**data)

        except Exception as e:
            logger.error(f"Scout failed: {e}")
            logger.info("Defaulting to General.")
            
        return DiscoveryResult(
            domain="General", 
            is_high_density=False, 
            nested_skeleton={
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "summary": {"type": "string", "description": "High-level summary"},
                    "key_entities": {
                        "type": "array", 
                        "description": "List of important entities",
                        "items": {"type": "string"}
                    }
                }
            },
            confidence=0.0
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
        extra_fields["page_references"] = (List[int], Field(default_factory=list))
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
