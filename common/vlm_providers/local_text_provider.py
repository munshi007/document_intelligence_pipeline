import logging
import os
import json
import re
from typing import Type, TypeVar, Optional, Dict, Any, List
import pydantic
import torch
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class LocalTextProvider(BaseVLMProvider):
    _model = None
    _tokenizer = None
    _current_path = None

    def __init__(self, model_name: str, **kwargs):
        self.model_path = model_name
        self._load_model()

    def _resolve_local_path(self, model_name: str) -> str:
        """Resolves a hub name to its local snapshot path if in offline mode."""
        if os.getenv("HF_HUB_OFFLINE") != "1":
            return model_name
        
        # If it's already a path, return it
        if os.path.exists(model_name):
            return model_name
            
        # Try to resolve from cache
        try:
            repo_id = model_name.replace("/", "--")
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache = os.path.join(cache_dir, f"models--{repo_id}")
            snapshots_dir = os.path.join(model_cache, "snapshots")
            
            if os.path.exists(snapshots_dir):
                snapshots = sorted(os.listdir(snapshots_dir))
                if snapshots:
                    resolved_path = os.path.join(snapshots_dir, snapshots[-1])
                    logger.info(f"Resolved TEXT model {model_name} to local snapshot: {resolved_path}")
                    return resolved_path
        except Exception:
            pass
            
        return model_name

    def _load_model(self):
        """Singleton-style model loader for purely TEXT Unsloth models (like Qwen Specialist)."""
        if LocalTextProvider._model is None or LocalTextProvider._current_path != self.model_path:
            try:
                from unsloth import FastLanguageModel
                import os
                
                is_offline = os.getenv("HF_HUB_OFFLINE") == "1"
                effective_path = self._resolve_local_path(self.model_path)
                
                logger.info(f"Loading TEXT model: {self.model_path} (Offline={is_offline})")
                
                LocalTextProvider._model, LocalTextProvider._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=effective_path,
                    max_seq_length=16384,
                    load_in_4bit=True,
                    device_map="cuda:0",
                    local_files_only=is_offline
                )
                FastLanguageModel.for_inference(LocalTextProvider._model)
                LocalTextProvider._current_path = self.model_path
            except Exception as e:
                logger.error(f"Failed to load local Unsloth Text model: {e}")
                raise

    @staticmethod
    def _extract_json_payload(text: str, expected_type: str = "object", return_candidates: bool = False):
        """
        Advanced JSON extractor that avoids picking empty lists '[]' when the schema 
        expects an object '{}'.
        """
        if not text:
            return (None, []) if return_candidates else None
        candidates = []
        fenced_candidates = []
        stripped = text.strip()

        # 1. Prioritize Fenced Blocks
        for match in re.finditer(r"```json\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            candidates.append(block)
            fenced_candidates.append(block)
        for match in re.finditer(r"```\s*(.*?)\s*```", stripped, re.DOTALL):
            block = match.group(1).strip()
            candidates.append(block)
            fenced_candidates.append(block)

        # Fast path: if fenced JSON parses directly, trust it and skip fragile salvage paths.
        for candidate in fenced_candidates:
            try:
                parsed = json.loads(candidate)
                if expected_type == "object" and isinstance(parsed, dict):
                    found_json = json.dumps(parsed)
                    if return_candidates:
                        return found_json, candidates
                    return found_json
                if expected_type == "list" and isinstance(parsed, list):
                    found_json = json.dumps(parsed)
                    if return_candidates:
                        return found_json, candidates
                    return found_json
                if isinstance(parsed, (dict, list)):
                    found_json = json.dumps(parsed)
                    if return_candidates:
                        return found_json, candidates
                    return found_json
            except Exception:
                continue

        # 2. Heuristic Bracket Search
        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj > first_obj:
            candidates.append(stripped[first_obj:last_obj + 1].strip())
            
        first_arr = stripped.find("[")
        last_arr = stripped.rfind("]")
        if first_arr != -1 and last_arr > first_arr:
            candidates.append(stripped[first_arr:last_arr + 1].strip())

        # 3. Strategy-Aware Selection: Iteratively try to find the longest valid JSON substring
        found_json = None
        for candidate in reversed(candidates):
            # A. Try to fix truncated JSON by finding the last closing brace
            last_brace = candidate.rfind("}")
            if last_brace != -1:
                candidate = candidate[:last_brace + 1]

            # B. Atomic Repair for structural hallucinations (Double Braces, etc.)
            # e.g. "identity": { { ... } -> "identity": { ... }
            candidate = re.sub(r'\{\s*\{', '{', candidate)
            candidate = re.sub(r'\}\s*\}', '}', candidate)

            # C. Atomic Repair for Hallucinated Escapes, Trailing Commas & Truncation
            # Common model bugs: \U for capital degrees, \u without digits, trailing commas, etc.
            candidate = re.sub(r'\\U', 'U', candidate) 
            candidate = re.sub(r'\\(?!(u[0-9a-fA-F]{4}|["\\/bfnrt]))', ' ', candidate)
            # Fix trailing commas: {"key": value, } -> {"key": value }
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)

            # Fix truncation: If it ends in a comma or a key/value, attempt to close it
            if not candidate.strip().endswith(('}', ']')):
                logger.warning("Detected potential JSON truncation. Attempting automatic closure.")
                # Simple balanced bracket salvage
                open_braces = candidate.count('{') - candidate.count('}')
                open_brackets = candidate.count('[') - candidate.count(']')
                
                # Strip trailing junk like commas or partial keys
                candidate = re.sub(r',\s*$', '', candidate.strip())
                candidate = re.sub(r'"[^"]*$', '', candidate)
                candidate = re.sub(r':[^:]*$', '', candidate)
                
                candidate += ']' * max(0, open_brackets)
                candidate += '}' * max(0, open_braces)

            try:
                parsed = json.loads(candidate)
                if expected_type == "object" and isinstance(parsed, dict):
                    found_json = json.dumps(parsed)
                    break
                if expected_type == "list" and isinstance(parsed, list):
                    found_json = json.dumps(parsed)
                    break
                if isinstance(parsed, (dict, list)):
                    found_json = json.dumps(parsed)
                    break
            except Exception:
                # C. Nuclear Fallback: Strip ALL backslashes and try one last time
                # This fixes the most stubborn hallucinated escapes (\u\U, \u1, etc.)
                try:
                    nuclear_candidate = candidate.replace("\\", " ")
                    parsed = json.loads(nuclear_candidate)
                    if isinstance(parsed, (dict, list)):
                        found_json = json.dumps(parsed)
                        break
                except Exception:
                    try:
                        # D. Hyper-Nuclear: Strip non-printable and hope for the best
                        sanitized = "".join(char for char in nuclear_candidate if char.isprintable() or char in "\n\r\t")
                        # Remove common JSON-breaking hallucinations
                        sanitized = sanitized.replace("...", " ")
                        parsed = json.loads(sanitized)
                        if isinstance(parsed, (dict, list)):
                            found_json = json.dumps(parsed)
                            break
                    except Exception as final_e:
                        # E. Nuclear Segmented Recovery: Harvest individual { ... } objects using stack-based bracket matching
                        try:
                            # Clean up the most common array-breakers
                            harvest_candidate = candidate.replace("...", "").replace(", ,", ",")
                            
                            # Nuclear Brace Harvester: Find every valid top-level or nested { } block
                            parsed_segments = []
                            stack = []
                            start_idx = -1
                            
                            for idx, char in enumerate(harvest_candidate):
                                if char == '{':
                                    stack.append(idx)
                                elif char == '}':
                                    if stack:
                                        start_pos = stack.pop()
                                        # Only harvest top-level children or independent segments
                                        # (Depth check ensures we don't accidentally grab tiny sub-objects while the 
                                        # parent object is still potentially valid, unless we are in a truncated state).
                                        segment = harvest_candidate[start_pos:idx+1]
                                        # Repair mid-stream dangling commas in the segment
                                        segment = re.sub(r',\s*([\]}])', r'\1', segment)
                                        try:
                                            # Validate the segment immediately
                                            p = json.loads(segment)
                                            if isinstance(p, dict):
                                                # Check if this segment is worth keeping (has hardware keys)
                                                if any(k in p for k in ["name", "product_name", "pins", "led"]):
                                                    parsed_segments.append(p)
                                        except:
                                            # Final-character sub-repair
                                            try:
                                                repaired = re.sub(r',\s*$', '', segment[:-1].strip()) + '}'
                                                p = json.loads(repaired)
                                                if any(k in p for k in ["name", "product_name", "pins", "led"]):
                                                    parsed_segments.append(p)
                                            except:
                                                continue
                            
                            if parsed_segments:
                                # Reconstruct the Universal Hardware container
                                reconstructed = {"identity": {}, "parameters": [], "connectors": [], "diagnostics": []}
                                for p in parsed_segments:
                                    if "name" in p and "value" in p: reconstructed["parameters"].append(p)
                                    elif "product_name" in p: reconstructed["identity"] = p
                                    elif "pins" in p: reconstructed["connectors"].append(p)
                                    elif "led" in p: reconstructed["diagnostics"].append(p)
                                
                                if reconstructed["parameters"] or reconstructed["identity"]:
                                    found_json = json.dumps(reconstructed)
                                    break
                        except Exception:
                            logger.error(f"Nuclear Recovery Failed: {final_e}")
                            continue
        
        if return_candidates:
            return found_json, candidates
        return found_json

    @staticmethod
    def _map_hallucinated_fields(partial: Dict[str, Any], response_model: Type) -> Dict[str, Any]:
        """Maps model output keys to schema field names and reshapes flat lists to nested objects."""
        schema_fields = list(response_model.model_fields.keys())
        model_keys = list(partial.keys())
        
        # 1. Synonym Mapping
        synonyms = {
            "art_no": ["articleNumber", "article_number"],
            "product_name": ["moduleName", "deviceName", "device_name"],
            "connectors": ["ports", "connections"],
            "pins": ["pinAssignments", "pinout"],
            "parameters": ["technicalData", "specifications"]
        }
        
        for m_key in model_keys:
            if m_key in schema_fields: continue
            if m_key in synonyms:
                for s_cand in synonyms[m_key]:
                    if s_cand in schema_fields:
                        if partial.get(s_cand) is None or partial.get(s_cand) == []:
                            logger.info(f"Synonym Mapping: '{m_key}' -> '{s_cand}'")
                            partial[s_cand] = partial.pop(m_key)
                            break

        # 2. Structural Reshaping (List -> Object)
        # If model returned a list for 'ports' but schema expects an object with specific sub-keys
        if "ports" in partial and isinstance(partial["ports"], list):
            port_field = response_model.model_fields.get("ports")
            if port_field:
                # Unwrap Optional/Union to get the actual model class
                ann = port_field.annotation
                if hasattr(ann, "__args__"):
                    ann = ann.__args__[0]
                
                if hasattr(ann, "model_fields"):
                    logger.info("Reshaping flat 'ports' list into structured object...")
                    original_list = partial.pop("ports")
                    new_ports = {}
                    new_pin_assignments = {}
                    sub_keys = ann.model_fields.keys()
                    
                    for item in original_list:
                        if not isinstance(item, dict): continue
                        name = str(item.get("name", "")).lower()
                        # Heuristic routing
                        target_key = None
                        if "input" in name: target_key = "inputPort"
                        elif "output" in name: target_key = "outputPort"
                        elif "multi" in name or "multifunctional" in name: target_key = "multifunctionalPort"
                        
                        if target_key and target_key in sub_keys:
                            # 1. Lift and Translate pins to pinAssignments if needed
                            if "pins" in item:
                                pins = item.pop("pins")
                                if isinstance(pins, list):
                                    # Translate pin sub-keys
                                    for p_item in pins:
                                        if not isinstance(p_item, dict): continue
                                        # Map model 'pin' or 'number' to 'pinNumber'
                                        if "pin" in p_item: p_item["pinNumber"] = p_item.pop("pin")
                                        if "number" in p_item: p_item["pinNumber"] = p_item.pop("number")
                                        # Map model 'signal' or 'assignment' to 'function'
                                        if "signal" in p_item: p_item["function"] = p_item.pop("signal")
                                        if "assignment" in p_item: p_item["function"] = p_item.pop("assignment")
                                    new_pin_assignments[target_key] = pins
                            
                            # 2. Translate model keys to schema keys
                            mapping = {
                                "type": "connectorType",
                                "gender": "connectorGender",
                                "coding": "connectorCoding"
                            }
                            for m_sub, s_sub in mapping.items():
                                if m_sub in item: item[s_sub] = item.pop(m_sub)
                            
                            # 3. Derive pinCount if missing
                            if target_key in new_pin_assignments:
                                item["pinCount"] = len(new_pin_assignments[target_key])
                            
                            new_ports[target_key] = item
                    
                    partial["ports"] = new_ports
                    if new_pin_assignments and "pinAssignments" in schema_fields:
                        partial["pinAssignments"] = new_pin_assignments

        # 3. Fuzzy/Substring Check (Final Pass)
        for m_key in list(partial.keys()):
            if m_key in schema_fields: continue
            for s_key in schema_fields:
                if (m_key in s_key or s_key in m_key) and len(m_key) > 4:
                    if partial.get(s_key) is None:
                        logger.info(f"Fuzzy Mapping: '{m_key}' -> '{s_key}'")
                        partial[s_key] = partial.pop(m_key)
                        break
        return partial

    @staticmethod
    def _normalize_common_schema_values(partial: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize frequent model typos before Pydantic validation.

        Keeps behavior conservative: only fix values that are very likely intent-preserving.
        """
        try:
            params = partial.get("parameters")
            if isinstance(params, list):
                # Standard mapping
                typo_map = {
                    "mechnaical": "mechanical",
                    "mechnical": "mechanical",
                    "mechanicalcal": "mechanical",
                    "electrial": "electrical",
                    "electricalal": "electrical",
                    "enviromental": "environmental",
                    "enviornmental": "environmental",
                    "logisitical": "logistical",
                    "logistic": "logistical",
                }
                valid_types = {"electrical", "mechanical", "environmental", "logistical"}
                
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    raw = p.get("param_type")
                    if isinstance(raw, str):
                        k = raw.strip().lower()
                        # Level 1: Standard typos
                        if k in typo_map:
                            p["param_type"] = typo_map[k]
                        # Level 2: Fuzzy matching for repetitive suffixes (e.g. 'mechanicalcal')
                        elif k not in valid_types:
                            for valid in valid_types:
                                if k.startswith(valid):
                                    p["param_type"] = valid
                                    break
        except Exception:
            pass
        return partial

    @staticmethod
    def _coerce_list_fields(partial: Dict[str, Any], response_model: Type) -> Dict[str, Any]:
        """Replace null with [] for all list-typed fields to prevent Pydantic validation failures.

        The extraction model frequently outputs ``null`` for list fields (e.g.
        ``"connectors": null``) when nothing was found.  Pydantic v2 rejects
        ``null`` for ``List[X]`` fields even when a ``default_factory=list`` is
        set.  This method repairs the dict in-place before validation.
        """
        import typing
        try:
            for field_name, field_info in response_model.model_fields.items():
                if partial.get(field_name) is None:
                    ann = field_info.annotation
                    origin = typing.get_origin(ann)
                    if origin is list:
                        partial[field_name] = []
                    elif origin is typing.Union:
                        for arg in typing.get_args(ann):
                            if typing.get_origin(arg) is list:
                                partial[field_name] = []
                                break
        except Exception:
            pass
        return partial

    @staticmethod
    def _coerce_type_mismatches(partial: Dict[str, Any], response_model: Type) -> Dict[str, Any]:
        """Coerce values whose types don't match the schema.

        Common case: model returns a list where schema expects a scalar, or
        a scalar where schema expects a list.  This prevents Pydantic from
        rejecting otherwise-correct extractions.
        """
        import typing
        try:
            for field_name, field_info in response_model.model_fields.items():
                if field_name not in partial or partial[field_name] is None:
                    continue
                val = partial[field_name]
                ann = field_info.annotation
                # Unwrap Optional[X] -> X
                origin = typing.get_origin(ann)
                if origin is typing.Union:
                    args = [a for a in typing.get_args(ann) if a is not type(None)]
                    if len(args) == 1:
                        ann = args[0]
                        origin = typing.get_origin(ann)

                # Case 1: schema expects scalar but model returned a list
                if origin is not list and isinstance(val, list):
                    if len(val) == 0:
                        partial[field_name] = None
                    elif len(val) == 1:
                        partial[field_name] = val[0]
                    else:
                        # Multiple values: keep as-is if schema is str (join), otherwise take first
                        if ann is str:
                            partial[field_name] = ", ".join(str(v) for v in val)
                        else:
                            # Store the list as a string representation so we don't lose data
                            partial[field_name] = val[0]
                            logger.info(f"Coerced list->scalar for '{field_name}': kept first of {len(val)} values")

                # Case 2: schema expects list but model returned a scalar
                if origin is list and not isinstance(val, list):
                    partial[field_name] = [val]
        except Exception as e:
            logger.warning(f"Type coercion warning: {e}")
        return partial

    @staticmethod
    def _append_jsonl(path: Optional[str], payload: Dict[str, Any]) -> None:
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_generation(self, full_prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": full_prompt}]
        inputs = LocalTextProvider._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            outputs = LocalTextProvider._model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=LocalTextProvider._tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.05,
            )
        prompt_len = inputs.shape[-1]
        generated = outputs[0][prompt_len:]
        return LocalTextProvider._tokenizer.decode(generated, skip_special_tokens=True)

    def _dereference_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict): return schema
        defs = schema.get("$defs", schema.get("definitions", {}))
        
        def resolve(node):
            if isinstance(node, list): return [resolve(i) for i in node]
            if not isinstance(node, dict): return node
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                if ref_key in defs:
                    return resolve(defs[ref_key].copy())
            return {k: resolve(v) for k, v in node.items() if k not in ["$defs", "definitions"]}
        return resolve(schema)

    def generate_structured(
        self, 
        image: Any, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False,
        **kwargs
    ) -> Optional[T]:
        if not LocalTextProvider._model:
            return None

        schema = self._dereference_schema(response_model.model_json_schema())
        
        full_prompt = f"""{prompt}

    You are the Librarian Extraction Specialist. 
    First, perform step-by-step reasoning about the document nodes within a `<thought>` block.
    Then, output exactly the final result as a valid JSON block that strictly matches this schema.
    
    ### CONSTRAINTS:
    1. Output a SINGLE valid JSON object.
    2. If no data is found for a specific field, return an empty array `[]`.
    3. If NO RELEVANT DATA is found in the entire set of nodes, return an empty object `{{}}` with default values, NOT an empty list `[]`.
    
SCHEMA:
{json.dumps(schema, indent=2)}
"""

        trace_dir = kwargs.get("trace_dir")
        trace_key = kwargs.get("trace_key", "batch")
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        def write_trace(filename: str, content: str) -> None:
            if not trace_dir:
                return
            out_path = os.path.join(trace_dir, f"{trace_key}_{filename}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        try:
            max_tokens = kwargs.get('max_tokens', kwargs.get('max_new_tokens', 4096))

            # Attempt 1: strict schema generation
            decoded = self._run_generation(full_prompt, max_tokens)
            write_trace("attempt1_raw.txt", decoded)
            json_payload, candidates = self._extract_json_payload(decoded, expected_type="object", return_candidates=True)
            if json_payload:
                try:
                    _p1_orig = json.loads(json_payload)
                    # Loop Breaker: If VLM returns a list when we expect an object, it's a hallucination.
                    if isinstance(_p1_orig, list) and not isinstance(response_model, list):
                        logger.warning(f"VLM returned LIST for master OBJECT {response_model.__name__}. Attempting wrapper salvage.")
                        # If list is empty, treat as empty dict
                        if not _p1_orig:
                            _p1_orig = {}
                        else:
                            # Heuristic: Find first list field in schema and put it there
                            _p1_orig = {"parameters": _p1_orig}
                    
                    _p1 = self._map_hallucinated_fields(_p1_orig, response_model)
                    _p1 = self._coerce_list_fields(_p1, response_model)
                    _p1 = self._coerce_type_mismatches(_p1, response_model)
                    _p1 = self._normalize_common_schema_values(_p1)
                    return response_model.model_validate(_p1)
                except Exception as ve:
                    logger.error(f"Pydantic Validation Error: {ve}")
                    self._append_jsonl(
                        os.path.join(trace_dir, "parse_failures.jsonl") if trace_dir else None,
                        {
                            "trace_key": trace_key,
                            "attempt": 1,
                            "error": str(ve),
                            "candidates": [c[:400] for c in candidates],
                        },
                    )

            # Attempt 2: repair prompt over raw output
            repair_prompt = f"""Repair the following model output into ONE valid JSON object matching this schema.
Return ONLY JSON with no commentary.

SCHEMA:
{json.dumps(schema, indent=2)}

RAW OUTPUT:
{decoded}
"""
            repaired = self._run_generation(repair_prompt, max_tokens)
            write_trace("attempt2_repair_raw.txt", repaired)
            repaired_payload, repaired_candidates = self._extract_json_payload(repaired, expected_type="object", return_candidates=True)
            if repaired_payload:
                try:
                    _p2 = self._map_hallucinated_fields(json.loads(repaired_payload), response_model)
                    _p2 = self._coerce_list_fields(_p2, response_model)
                    _p2 = self._coerce_type_mismatches(_p2, response_model)
                    _p2 = self._normalize_common_schema_values(_p2)
                    return response_model.model_validate(_p2)
                except Exception as ve:
                    logger.warning(f"Repair validation failed: {ve}")
                    self._append_jsonl(
                        os.path.join(trace_dir, "parse_failures.jsonl") if trace_dir else None,
                        {
                            "trace_key": trace_key,
                            "attempt": 2,
                            "error": str(ve),
                            "candidates": [c[:400] for c in repaired_candidates],
                        },
                    )

            # Attempt 3: partial salvage with defaults
            candidate_payload = repaired_payload or json_payload
            if candidate_payload:
                try:
                    partial = json.loads(candidate_payload)
                    for field_name, field_info in response_model.model_fields.items():
                        if field_name not in partial:
                            if field_info.default is not None:
                                partial[field_name] = field_info.default
                            elif field_info.default_factory is not None:
                                partial[field_name] = field_info.default_factory()
                    partial = self._coerce_list_fields(partial, response_model)
                    partial = self._map_hallucinated_fields(partial, response_model)
                    partial = self._coerce_type_mismatches(partial, response_model)
                    partial = self._normalize_common_schema_values(partial)
                    try:
                        return response_model.model_validate(partial)
                    except Exception:
                        # FINAL SAFETY NET: use model_construct to never lose data
                        logger.warning(f"Final salvage: using model_construct for {response_model.__name__}")
                        return response_model.model_construct(**partial)
                except Exception as ve:
                    self._append_jsonl(
                        os.path.join(trace_dir, "parse_failures.jsonl") if trace_dir else None,
                        {
                            "trace_key": trace_key,
                            "attempt": 3,
                            "error": str(ve),
                            "note": "partial_salvage_failed",
                        },
                    )

            logger.warning("Local Text Provider: no valid JSON extracted from response after retries")
            return None
        except Exception as e:
            logger.error(f"Local Text Provider Error: {e}")
            return None

    def generate(self, image: Any, prompt: str, is_complex: bool = False, **kwargs) -> str:
        if not LocalTextProvider._model: return "Error: Model uninitialized"
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = LocalTextProvider._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            outputs = LocalTextProvider._model.generate(
                inputs, max_new_tokens=4096, use_cache=True, do_sample=False
            )
            return LocalTextProvider._tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Local Text Error: {e}")
            return f"Error: {e}"
