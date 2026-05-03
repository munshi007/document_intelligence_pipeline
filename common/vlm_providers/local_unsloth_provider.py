import logging
import os
import json
import re
from typing import Type, TypeVar, Optional, Dict, Any, List
from PIL import Image
import pydantic
import torch
from .base import BaseVLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=pydantic.BaseModel)

class LocalUnslothProvider(BaseVLMProvider):
    _model = None
    _tokenizer = None
    _current_path = None

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
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
                    logger.info(f"Resolved {model_name} to local snapshot: {resolved_path}")
                    return resolved_path
        except Exception:
            pass
            
        return model_name

    def _load_model(self):
        """Singleton-style model loader for Unsloth models."""
        if LocalUnslothProvider._model is None or LocalUnslothProvider._current_path != self.model_path:
            try:
                from unsloth import FastVisionModel
                import os
                
                is_offline = os.getenv("HF_HUB_OFFLINE") == "1"
                effective_path = self._resolve_local_path(self.model_path)
                
                logger.info(f"Loading VLM: {self.model_path} (Offline={is_offline})")
                
                LocalUnslothProvider._model, LocalUnslothProvider._tokenizer = FastVisionModel.from_pretrained(
                    model_name=effective_path,
                    load_in_4bit=True,
                    device_map="cuda:0",
                    trust_remote_code=True,
                    local_files_only=is_offline  # CRITICAL: Force local load if offline
                )
                FastVisionModel.for_inference(LocalUnslothProvider._model)
                LocalUnslothProvider._current_path = self.model_path
            except Exception as e:
                import os
                if os.getenv("HF_HUB_OFFLINE") == "1":
                    logger.error(f"Vision Model: Failed to load from local cache! Ensure {self.model_path} is downloaded. Error: {e}")
                else:
                    logger.error(f"Failed to load local Unsloth model: {e}")
                LocalUnslothProvider._model = None

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[str]:
        """Extract last valid JSON object from model output, handling hallucinations."""
        if not text:
            return None

        # Clean text from common artifacts
        text = text.strip()
        
        # 1. Look for markdown code blocks (standard for Llama-3)
        fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        fences = re.findall(fence_pattern, text, re.DOTALL)
        
        # 2. Search for raw { } if no fences found
        raw_pattern = r"(\{.*\})"
        raw_matches = re.findall(raw_pattern, text, re.DOTALL)
        
        candidates = fences + raw_matches
        
        # Validate from the bottom up (the final output is usually the most relevant)
        for candidate in reversed(candidates):
            candidate = candidate.strip()
            try:
                # Basic cleaning
                candidate = re.sub(r'//[^\n]*', '', candidate)  # Strip inline comments
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate) # Strip trailing commas
                
                parsed = json.loads(candidate)
                
                # Filter out schema definitions accidentally echoed
                if isinstance(parsed, dict) and "properties" in parsed and "type" in parsed:
                    continue
                    
                return json.dumps(parsed)
            except Exception:
                continue

        return None

    @staticmethod
    def _build_example_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build a concrete example JSON object from a dereferenced schema.
        Shows the model WHAT to output rather than HOW to describe it."""
        example = {}
        props = schema.get("properties", {})
        for field_name, field_def in props.items():
            field_type = field_def.get("type", "string")
            if "enum" in field_def:
                example[field_name] = field_def["enum"][0]
            elif "anyOf" in field_def:
                # Optional field — pick first non-null type
                for opt in field_def["anyOf"]:
                    if opt.get("type") != "null":
                        example[field_name] = {"string": "...", "boolean": False, "integer": 0, "number": 0.0}.get(opt.get("type"), None)
                        break
                else:
                    example[field_name] = None
            elif field_type == "array":
                example[field_name] = []
            elif field_type == "boolean":
                example[field_name] = False
            elif field_type == "integer":
                example[field_name] = 0
            elif field_type == "number":
                example[field_name] = 0.0
            else:
                example[field_name] = "..."
        return example

    def generate_structured(
        self, 
        image: Image.Image, 
        prompt: str, 
        response_model: Type[T],
        is_complex: bool = False,
        **kwargs
    ) -> Optional[T]:
        if not LocalUnslothProvider._model:
            return None

        schema = self._dereference_schema(response_model.model_json_schema())
        example = self._build_example_from_schema(schema)
        
        # Build a concise field-description guide
        field_lines = []
        for field_name, field_def in schema.get("properties", {}).items():
            desc = field_def.get("description", "")
            field_lines.append(f'  "{field_name}": <value>  // {desc}')
        fields_doc = "\n".join(field_lines)

        full_prompt = f"""{prompt}

CRITICAL: Respond ONLY with a single JSON object inside markdown backticks. 
Identify all key attributes and nested fields based on the description below.

Schema and Descriptions:
{fields_doc}

Example Structure:
```json
{json.dumps(example, indent=2)}
```
Now, output the FINAL JSON following the EXACT pattern above:"""

        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": full_prompt})

        messages = [
            {"role": "user", "content": content}
        ]

        try:
            input_text = LocalUnslothProvider._tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            tokenization_kwargs = {"text": input_text, "add_special_tokens": False, "return_tensors": "pt"}
            if image is not None:
                tokenization_kwargs["images"] = image
                
            inputs = LocalUnslothProvider._tokenizer(**tokenization_kwargs).to("cuda")


            # Use longer max_tokens for general extraction (summaries, tables)
            max_tokens = kwargs.get('max_tokens', kwargs.get('max_new_tokens', 1024))

            outputs = LocalUnslothProvider._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                use_cache=True,
                temperature=0.1,
                do_sample=False
            )

            prompt_len = inputs["input_ids"].shape[-1]
            generated = outputs[0][prompt_len:]
            decoded = LocalUnslothProvider._tokenizer.decode(generated, skip_special_tokens=True)
            logger.debug(f"VLM raw [{response_model.__name__}]: {decoded[:400]}")

            json_payload = self._extract_json_payload(decoded)
            if json_payload:
                try:
                    return response_model.model_validate_json(json_payload)
                except Exception as val_err:
                    logger.warning(f"VLM validation failed for {response_model.__name__}: {val_err}")
                    # Attempt partial coercion using model defaults
                    try:
                        partial = json.loads(json_payload)
                        for field_name, field_info in response_model.model_fields.items():
                            if field_name not in partial:
                                if field_info.default is not None:
                                    partial[field_name] = field_info.default
                                elif field_info.default_factory is not None:
                                    partial[field_name] = field_info.default_factory()
                        return response_model.model_construct(**partial)
                    except Exception:
                        pass

            logger.warning(f"VLM: no valid JSON for {response_model.__name__}")
            # Parse raw response
            response = decoded
            logger.debug(
                "Local Unsloth Provider raw response (len=%s): %s",
                len(response),
                response[:1000],
            )
            return None
        except Exception as e:
            logger.error(f"Local Unsloth Provider Error: {e}")
            return None

    def _dereference_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves $ref pointers in a JSON schema recursively to provide a flat structure to the model."""
        if not isinstance(schema, dict):
            return schema
            
        defs = schema.get("$defs", schema.get("definitions", {}))
        
        def resolve(node):
            if isinstance(node, list):
                return [resolve(i) for i in node]
            if not isinstance(node, dict):
                return node
                
            if "$ref" in node:
                ref_path = node["$ref"]
                ref_key = ref_path.split("/")[-1]
                if ref_key in defs:
                    # Return the resolved definition but recursively resolve it too
                    resolved = defs[ref_key].copy()
                    return resolve(resolved)
            
            return {k: resolve(v) for k, v in node.items() if k not in ["$defs", "definitions"]}
            
        return resolve(schema)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        is_complex: bool = False,
        **kwargs
    ) -> str:
<<<<<<< HEAD
        if LocalUnslothProvider._model is None:
            return "Error: Local model not initialized"

        try:
            # Consistent with generate_structured loop
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            input_text = LocalUnslothProvider._tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            inputs = LocalUnslothProvider._tokenizer(
                image, 
                input_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to("cuda")

            outputs = LocalUnslothProvider._model.generate(
                **inputs,
                max_new_tokens=4096,
=======
        if LocalUnslothProvider._model is None or LocalUnslothProvider._tokenizer is None:
            return "Error: Local model not initialized"

        try:
            content = []
            if image is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "user", "content": content}
            ]

            input_text = LocalUnslothProvider._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            tokenization_kwargs = {"text": input_text, "add_special_tokens": False, "return_tensors": "pt"}
            if image is not None:
                tokenization_kwargs["images"] = image
                
            inputs = LocalUnslothProvider._tokenizer(**tokenization_kwargs).to("cuda")


            # Extract max tokens from kwargs
            max_tokens = kwargs.get('max_tokens', kwargs.get('max_new_tokens', 4096))

            output = LocalUnslothProvider._model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
                use_cache=True,
                temperature=0.1,
                do_sample=False
            )
<<<<<<< HEAD
            
            # Use tokenizer for decoding
            result = LocalUnslothProvider._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
=======

            # Safely decode and remove the prompt from the response
            full_text = LocalUnslothProvider._tokenizer.decode(output[0], skip_special_tokens=True)
            # Find the start of the response after the prompt text
            if prompt in full_text:
                return full_text.split(prompt)[-1].strip()
            return full_text.strip()
>>>>>>> 49e79bc (docs: update README with detailed instructions and benchmarks; chore: finalize v3 pipeline)
            
        except Exception as e:
            logger.error(f"Local Unsloth Provider Error: {e}")
            return f"Error: {e}"
