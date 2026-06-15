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

    def _resolve_pinned_revision(self) -> Optional[str]:
        """If model_path matches the canonical manifest entry, return its pinned
        revision SHA so we fetch the exact snapshot used for thesis results."""
        try:
            from common import model_registry
            spec = model_registry.get("librarian_qwen")
            if self.model_path == spec.repo_id:
                return spec.revision
        except Exception:
            pass
        return None

    def _load_model(self):
        """Singleton-style model loader for purely TEXT Unsloth models (like Qwen Specialist)."""
        if LocalTextProvider._model is None or LocalTextProvider._current_path != self.model_path:
            try:
                from unsloth import FastLanguageModel
                import os

                is_offline = os.getenv("HF_HUB_OFFLINE") == "1"
                effective_path = self._resolve_local_path(self.model_path)
                pinned_revision = self._resolve_pinned_revision()

                if pinned_revision:
                    logger.info(f"Loading TEXT model: {self.model_path}@{pinned_revision[:10]} (Offline={is_offline})")
                else:
                    logger.info(f"Loading TEXT model: {self.model_path} (Offline={is_offline})")

                load_kwargs = dict(
                    model_name=effective_path,
                    max_seq_length=16384,
                    load_in_4bit=True,
                    device_map="cuda:0",
                    local_files_only=is_offline,
                )
                if pinned_revision and not is_offline:
                    load_kwargs["revision"] = pinned_revision

                LocalTextProvider._model, LocalTextProvider._tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                FastLanguageModel.for_inference(LocalTextProvider._model)
                LocalTextProvider._current_path = self.model_path
            except Exception as e:
                logger.error(f"Failed to load local Unsloth Text model: {e}")
                raise

    # A bare `...` / `…` standing where a JSON value would be — the model
    # abbreviating a long list ("and so on"). Structural-context lookarounds
    # keep ellipses inside string values untouched.
    _LIST_ABBREV_RE = re.compile(r'(?<=[\[{,])\s*(?:\.\.\.|…)\s*,?(?=\s*[\[{"\]},])')

    @staticmethod
    def _has_list_abbreviation(text: str) -> bool:
        """True when the model output abbreviates a list with a bare ellipsis."""
        return bool(text and LocalTextProvider._LIST_ABBREV_RE.search(text))

    @staticmethod
    def _strip_list_abbreviations(text: str) -> str:
        """Remove `...` / `…` abbreviation tokens from malformed JSON.

        Without this, the dangling `{` the abbreviation leaves behind makes
        downstream repairs swallow every key that follows the abbreviated
        list (observed: `grand_total` absorbed into a zombie list item and
        dropped). Only used on the repair path — never on JSON that already
        parses.
        """
        out = LocalTextProvider._LIST_ABBREV_RE.sub('', text)
        # Drop a dangling `{` the abbreviation left with no body before `]`.
        out = re.sub(r'(?:,\s*)?\{\s*(?=\])', '', out)
        return out

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
            # A0. Strip list-abbreviation ellipses BEFORE any bracket surgery —
            # they make every brace-counting repair below miscount.
            candidate = LocalTextProvider._strip_list_abbreviations(candidate)

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
            # Fix mixed-quote keys: "currency': "USD"  ->  "currency": "USD"
            # Observed on Qwen logistics extractions; one mistyped char on a
            # 6-charge JSON kills the whole parse. Only touches keys (chars
            # before `:`), so apostrophes inside string values are safe.
            candidate = re.sub(r'"([^"\n]+?)\'(\s*:)', r'"\1"\2', candidate)

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
                        # All per-candidate repairs (A–D) failed for this
                        # candidate. Move on; the schema-agnostic json_repair
                        # fallback below the loop is the final salvage and makes
                        # no field-name assumptions (no hardcoding).
                        logger.debug(f"Per-candidate JSON repair failed: {final_e}")
                        continue

        # E. Schema-agnostic structural repair (last resort, only if nothing
        # above produced valid JSON). json_repair makes no field-name
        # assumptions, so it works for any discovered schema. It fixes the
        # failures the bracket-balancing rules can't: a list left unclosed
        # *mid-structure* before the next object key (greedy decoder drops the
        # `]`), and runaway truncation.
        if found_json is None:
            try:
                from json_repair import repair_json
                # Drop any <thought>/prose preamble by starting at the first
                # structural char; json_repair handles the rest.
                obj_start = stripped.find("{")
                arr_start = stripped.find("[")
                starts = [i for i in (obj_start, arr_start) if i != -1]
                repair_src = stripped[min(starts):] if starts else stripped
                # Abbreviation ellipses make json_repair absorb every key
                # after the abbreviated list into a zombie list item.
                repair_src = LocalTextProvider._strip_list_abbreviations(repair_src)
                repaired_obj = repair_json(repair_src, return_objects=True)
                ok = (
                    (expected_type == "object" and isinstance(repaired_obj, dict))
                    or (expected_type == "list" and isinstance(repaired_obj, list))
                    or isinstance(repaired_obj, (dict, list))
                )
                if ok and repaired_obj:
                    logger.warning("Recovered malformed JSON via json_repair fallback.")
                    found_json = json.dumps(repaired_obj)
            except Exception as repair_err:
                logger.error(f"json_repair fallback failed: {repair_err}")

        if return_candidates:
            return found_json, candidates
        return found_json

    @staticmethod
    def _map_hallucinated_fields(partial: Dict[str, Any], response_model: Type) -> Dict[str, Any]:
        """Re-key model output whose keys drifted from the schema field names.

        Schema-agnostic: the only knowledge used is the target model's own
        ``model_fields`` -- no per-document field names or shapes are hardcoded.
        When the model emits a key that is a substring of a schema field (or
        vice-versa), e.g. ``parameters`` -> ``technical_parameters``, or a
        decode typo like ``connectorsors`` -> ``connectors``, and the real
        field is still empty, move the value onto the schema key so it is not
        dropped during validation.
        """
        schema_fields = list(response_model.model_fields.keys())
        for m_key in list(partial.keys()):
            if m_key in schema_fields:
                continue
            for s_key in schema_fields:
                if (m_key in s_key or s_key in m_key) and len(m_key) > 4:
                    if partial.get(s_key) is None:
                        logger.info(f"Fuzzy Mapping: '{m_key}' -> '{s_key}'")
                        partial[s_key] = partial.pop(m_key)
                        break
        return partial

    @staticmethod
    def _first_list_field(response_model: Type) -> Optional[str]:
        """Name of the first list-typed field in the schema, or None.

        Used to route a bare top-level list (model dropped the wrapper
        object) into the right field without hardcoding a field name.
        """
        import typing
        for field_name, field_info in response_model.model_fields.items():
            ann = field_info.annotation
            origin = typing.get_origin(ann)
            if origin is list:
                return field_name
            if origin is typing.Union:
                for arg in typing.get_args(ann):
                    if typing.get_origin(arg) is list:
                        return field_name
        return None

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
    def _list_item_model(annotation: Any) -> Optional[Type]:
        """Return the BaseModel item type of a ``List[ItemModel]`` annotation.

        Unwraps ``Optional[List[ItemModel]]`` too. Returns None for scalar
        lists (``List[str]``) or non-list fields. Schema-agnostic helper.
        """
        import typing
        origin = typing.get_origin(annotation)
        if origin is typing.Union:
            for arg in typing.get_args(annotation):
                model = LocalTextProvider._list_item_model(arg)
                if model is not None:
                    return model
            return None
        if origin is list:
            args = typing.get_args(annotation)
            if args:
                a0 = args[0]
                if isinstance(a0, type) and issubclass(a0, pydantic.BaseModel):
                    return a0
        return None

    @staticmethod
    def _realign_leftover_keys(item: Dict[str, Any], item_fields: set, required: List[str]) -> Dict[str, Any]:
        """Map an item's off-schema keys onto still-missing required fields.

        When a flattened item carries the real value under a drifted key
        (e.g. ``param`` instead of ``name``), move the best leftover string
        onto a missing required field. Also normalises a numeric ``min/max``
        range into an empty required ``value`` when the item model defines
        that triad. Gated entirely on the item model's own field names — no
        per-document or per-field hardcoding.
        """
        leftover = [k for k in list(item.keys()) if k not in item_fields]
        for f in required:
            if item.get(f) not in (None, ""):
                continue
            for k in list(leftover):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    item[f] = item.pop(k)
                    leftover.remove(k)
                    break
        # Range -> value normalisation for parameter-like {min,max,value} triads.
        if {"value", "min_value", "max_value"} <= item_fields and item.get("value") in (None, ""):
            mn, mx = item.get("min_value"), item.get("max_value")
            if mn not in (None, "") and mx not in (None, ""):
                item["value"] = f"{mn}...{mx}"
            elif mn not in (None, ""):
                item["value"] = str(mn)
            elif mx not in (None, ""):
                item["value"] = str(mx)
        return item

    @staticmethod
    def _unwrap_grouped_list_items(partial: Dict[str, Any], response_model: Type) -> Dict[str, Any]:
        """Flatten group-nested list items into the flat item list.

        The extraction model sometimes emits a *grouped* shape for a
        ``List[ItemModel]`` field, wrapping the real rows in a sub-list::

            "parameters": [{"name": "Voltage", "values": [ {..}, {..}, .. ]}]

        Pydantic keeps only the wrapper and silently drops the inner array,
        which destroyed recall (34 params -> 1 on the M12 datasheet). Detect
        a wrapper whose nested list (under an OFF-schema key) overlaps the
        item schema better than the wrapper's own keys, and flatten the
        inner rows up. Schema-driven: the only knowledge used is the item
        model's own field names, so declared nested fields (e.g. a
        connector's ``pins``) are never touched.
        """
        if not isinstance(partial, dict):
            return partial
        for field_name, field_info in response_model.model_fields.items():
            seq = partial.get(field_name)
            if not isinstance(seq, list) or not seq:
                continue
            item_model = LocalTextProvider._list_item_model(field_info.annotation)
            if item_model is None:
                continue
            item_fields = set(item_model.model_fields.keys())
            required = [n for n, fi in item_model.model_fields.items() if fi.is_required()]

            def _overlap(d: Any) -> int:
                return sum(1 for k in d if k in item_fields) if isinstance(d, dict) else 0

            flat: List[Any] = []
            changed = False
            for el in seq:
                if not isinstance(el, dict):
                    flat.append(el)
                    continue
                direct = _overlap(el)
                best = None  # (inner_items, overlap)
                for k, v in el.items():
                    if k in item_fields:
                        continue  # declared nested field — leave it alone
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        nov = max(_overlap(x) for x in v)
                        if nov > direct and (best is None or nov > best[1]):
                            best = (v, nov)
                if best is None:
                    flat.append(el)
                    continue
                # Carry the wrapper's own valid scalar fields down as defaults.
                carry = {k: v for k, v in el.items()
                         if k in item_fields and not isinstance(v, (list, dict)) and v not in (None, "")}
                for inner in best[0]:
                    merged = dict(inner)
                    LocalTextProvider._realign_leftover_keys(merged, item_fields, required)
                    for ck, cv in carry.items():
                        if merged.get(ck) in (None, ""):
                            merged[ck] = cv
                    flat.append(merged)
                changed = True
            if changed:
                logger.info(
                    f"Unwrapped group-nested items in '{field_name}': "
                    f"{len(seq)} wrapper(s) -> {len(flat)} flat item(s)."
                )
                partial[field_name] = flat
        return partial

    @staticmethod
    def _tolerant_list_salvage(payload: Dict[str, Any], response_model: Type[T]) -> Optional[T]:
        """Validate, dropping only the individual list items that fail.

        A single malformed row (e.g. a bad nested ``page_number``) otherwise
        sinks the whole document's recall, because ``model_validate`` is
        all-or-nothing. Here each list field's items are validated one by one;
        the valid ones are kept and the object is re-validated. Returns the
        validated model, or None if even the trimmed object will not validate.
        """
        if not isinstance(payload, dict):
            return None
        try:
            return response_model.model_validate(payload)
        except Exception:
            pass
        import copy
        trimmed = copy.deepcopy(payload)
        dropped = 0
        for field_name, field_info in response_model.model_fields.items():
            seq = trimmed.get(field_name)
            item_model = LocalTextProvider._list_item_model(field_info.annotation)
            if item_model is None or not isinstance(seq, list):
                continue
            kept = []
            for it in seq:
                try:
                    item_model.model_validate(it)
                    kept.append(it)
                except Exception:
                    dropped += 1
            trimmed[field_name] = kept
        try:
            model = response_model.model_validate(trimmed)
            if dropped:
                logger.warning(f"Tolerant salvage kept the record by dropping {dropped} invalid list item(s).")
            return model
        except Exception:
            return None

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
            # repetition_penalty=1.15 guards against the degenerate-loop
            # failure on long technical docs (e.g. Table_Test_1.pdf, where
            # Qwen got stuck alternating "Operating voltage" and "Current
            # consumption" entries until max_new_tokens was exhausted).
            # Initial fix also added no_repeat_ngram_size=8 but that turned
            # out too aggressive: docs with many similar nested objects
            # (e.g. Super_Complex_2.pdf with 3 connector dicts sharing the
            # same field layout) had the n-gram block prevent legitimate
            # `"name": "..."` -> `"type": "..."` sibling-key sequences,
            # producing malformed double-colon output. Penalty alone is
            # sufficient.
            outputs = LocalTextProvider._model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=LocalTextProvider._tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.15,
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

            # Anti-abbreviation regeneration: greedy decoding deterministically
            # abbreviates long arrays with a bare `...` element on some docs,
            # silently costing every item after it. Regenerate once with an
            # explicit completeness order; keep the original output if the
            # retry still abbreviates (the salvage path strips the ellipsis).
            if self._has_list_abbreviation(decoded):
                logger.warning(
                    "Attempt 1 abbreviated a list with '...'; regenerating with anti-abbreviation instruction."
                )
                anti_abbrev = (
                    "\n\nCRITICAL: A previous attempt abbreviated a long list with an "
                    "ellipsis ('...'). NEVER do this. Write out EVERY list item in "
                    "full, even when items are long, similar, or repetitive."
                )
                regenerated = self._run_generation(full_prompt + anti_abbrev, max_tokens)
                write_trace("attempt1b_no_abbrev_raw.txt", regenerated)
                if regenerated and not self._has_list_abbreviation(regenerated):
                    decoded = regenerated
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
                            # Route the bare list into the schema's first
                            # list-typed field (schema-agnostic; no hardcoded
                            # field name). This also recovers data on
                            # non-hardware schemas (e.g. a logistics doc whose
                            # list belongs in 'charges'), which the old
                            # hardcoded 'parameters' key silently dropped.
                            _lf = self._first_list_field(response_model)
                            _p1_orig = {_lf: _p1_orig} if _lf else {}
                    
                    _p1 = self._map_hallucinated_fields(_p1_orig, response_model)
                    _p1 = self._coerce_list_fields(_p1, response_model)
                    _p1 = self._unwrap_grouped_list_items(_p1, response_model)
                    _p1 = self._coerce_type_mismatches(_p1, response_model)
                    return response_model.model_validate(_p1)
                except Exception as ve:
                    logger.error(f"Pydantic Validation Error: {ve}")
                    # Before the lossy repair *regeneration*, salvage this
                    # already-parsed (often rich) payload by dropping only the
                    # individual list items that fail validation — one bad row
                    # must not cost the whole document's recall.
                    salvaged = self._tolerant_list_salvage(_p1, response_model)
                    if salvaged is not None:
                        logger.warning("Recovered attempt-1 payload via tolerant per-item salvage.")
                        return salvaged
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
                    _p2 = self._unwrap_grouped_list_items(_p2, response_model)
                    _p2 = self._coerce_type_mismatches(_p2, response_model)
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
                    partial = self._unwrap_grouped_list_items(partial, response_model)
                    partial = self._coerce_type_mismatches(partial, response_model)
                    try:
                        return response_model.model_validate(partial)
                    except Exception:
                        salvaged = self._tolerant_list_salvage(partial, response_model)
                        if salvaged is not None:
                            return salvaged
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
