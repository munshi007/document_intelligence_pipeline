# Universal Schema Extraction - Implementation Checklist

## Objective
Make one reliable pipeline that supports:
1. PDF -> Markdown/Manifest/Graph
2. Any schema passed at runtime
3. Strong extraction with explicit failure diagnostics (no silent empty outputs)

---

## Sprint Structure (10 Days)

## Day 1 - Canonical Runner and Artifact Contracts

### File: run_v3.py
### Changes
1. Add CLI args:
- `--schema_mode` with values `auto|domain|explicit`
- `--schema_path` for explicit extraction schema JSON
- `--save_debug_traces` flag

2. Standardize artifact names:
- Discovery routing output: `<doc>_schema_discovery.json`
- Final extraction schema used: `<doc>_target_schema.json`
- Final extraction output: `<doc>_extraction_result.json`

3. Ensure `--schema_mode explicit` does not depend on domain audit.

### Functions to touch
- `main()` argument parser block
- Step 4 extraction block (`if args.extract:`)

### Acceptance
- Running with `--schema_mode explicit --schema_path <file>` bypasses domain routing.
- All 3 schema artifacts are produced with unambiguous names.

---

## Day 2 - Discovery vs Extraction Schema Separation

### File: extractor/schema_engine.py
### Changes
1. Keep `audit_document()` as routing-only.
2. Keep `get_discovery_schema()` but ensure it only returns:
- `domain`
- `active_modules`
- `recommended_schema_family`

3. Add helper:
- `build_target_schema_contract(discovery, explicit_schema=None)`

### File: extractor/schema_registry.py
### Changes
1. Add registry for schema families:
- `hardware_v1`
- `invoice_v1`
- `contract_v1`
- `academic_v1`
- `general_v1`

2. Add function:
- `get_schema_family(domain: str) -> str`
- `get_schema_model(schema_family: str)`

### Acceptance
- Discovery JSON no longer looks like final extraction schema.
- Target schema artifact is available before extraction starts.

---

## Day 3 - Add Invoice and Generic Business Schemas

### File: extractor/schema_definitions.py
### Changes
1. Keep existing hardware model.
2. Add `LibrarianInvoiceRecord` model with strict core fields:
- `invoice_number`
- `invoice_date`
- `supplier`
- `recipient`
- `currency`
- `line_items[]` (description, qty, unit_price, amount)
- `subtotal`
- `vat[]`
- `total`
- `page_references`
- `confidence_score`

3. Add `LibrarianBusinessRecord` for non-invoice corporate docs.

### Acceptance
- Corporate docs can map to invoice/business schema families instead of generic fallback.

---

## Day 4 - Universal Extractor Interface

### File: extractor/agent.py
### Changes
1. Update `extract_structured()` to accept:
- `target_schema_name`
- `target_schema_json` (optional)
- `trace_context` (optional)

2. Replace default-empty-success behavior:
- Current behavior returns `response_model()` when no data.
- New behavior must return a structured failure object or raise and be handled upstream.

3. Add extraction quality scoring:
- `populated_fields_count`
- `required_fields_missing`
- `batch_success_count`

### Functions to touch
- `extract_structured()`
- `_synthesize_results()`

### Acceptance
- No more silent all-null output marked as success.

---

## Day 5 - JSON Reliability Hardening

### File: common/vlm_providers/local_text_provider.py
### Changes
1. Strengthen `generate_structured()` with retry ladder:
- Attempt 1: strict schema prompt
- Attempt 2: JSON-only repair prompt
- Attempt 3: partial salvage with explicit `_errors`

2. In `_extract_json_payload()`:
- keep best valid object
- enforce top-level object requirement
- include optional debug return of raw candidates

3. Add trace logging hook:
- save raw model output when parsing fails

### Acceptance
- Parse failures produce diagnostic payload and trace files.
- Failures no longer collapse directly to `None` without context.

---

## Day 6 - Chunking Integration for Extraction Context

### File: extractor/agent.py
### Changes
1. Replace raw char slicing (`max_batch_size=4000`) as primary strategy.
2. Add node-aware batching input:
- Use graph nodes grouped by semantic boundaries
- Preserve table atomicity

### File: chunker/graph_builder.py
### Changes
1. Expose helper to emit extraction-ready batches:
- `to_extraction_batches(max_chars, preserve_atomic=True)`

### File: processors/layout_chunker.py
### Changes
1. Add chunk metadata fields:
- `section_path`
- `region_ids`
- `contains_table`
- `linked_region_ids`

### Acceptance
- No batch splits in the middle of a table.
- Extractor consumes semantically bounded chunks.

---

## Day 7 - Observability and Debug Traces

### New folder
- `output/v3/debug_traces/`

### New files per run
- `<doc>_batch_trace.jsonl`
- `<doc>_parse_failures.jsonl`
- `<doc>_validation_summary.json`

### Data to log per batch
- batch_id
- schema_family
- prompt_hash
- input_chars
- raw_output_path
- parse_status
- validation_errors
- populated_fields_count

### Acceptance
- Every extraction failure can be diagnosed without rerunning.

---

## Day 8 - Evaluator for Multi-Document Families

### New file: extractor/evaluation.py
### Implement
1. Schema validity rate
2. Required-field completion rate
3. Non-empty extraction rate
4. Grounded reference rate (`page_references` non-empty)
5. Retry frequency

### CLI integration
- add `--evaluate` flag in `run_v3.py`

### Acceptance
- Numeric scorecard produced for each run.

---

## Day 9 - Regression Suite

### New folder: tests/extraction/
### Add tests
1. `test_invoice_schema_path.py`
2. `test_hardware_schema_path.py`
3. `test_explicit_schema_mode.py`
4. `test_parse_failure_diagnostics.py`

### Core assertions
- output is schema-valid
- output is not default empty unless explicitly marked failed
- debug traces exist on failure

### Acceptance
- Regression tests pass on at least 3 doc families.

---

## Day 10 - Documentation and Final CLI Modes

### File: README.md
### Add sections
1. Schema modes:
- `auto`
- `domain`
- `explicit`

2. Artifact map with filenames
3. Failure diagnostics interpretation guide
4. Example commands for all modes

### Acceptance
- A new user can run any schema path without code edits.

---

## File-by-File Patch Order (Exact Sequence)
1. `run_v3.py`
2. `extractor/schema_registry.py`
3. `extractor/schema_definitions.py`
4. `extractor/schema_engine.py`
5. `extractor/agent.py`
6. `common/vlm_providers/local_text_provider.py`
7. `chunker/graph_builder.py`
8. `processors/layout_chunker.py`
9. `extractor/evaluation.py` (new)
10. `tests/extraction/*` (new)
11. `README.md`

---

## Command Plan (Use These as Milestones)

### Milestone A - Explicit schema path works
```bash
python run_v3.py data/sample-invoice.pdf \
  --extract \
  --schema_mode explicit \
  --schema_path schema_sample.json \
  --output_dir output/v3
```

### Milestone B - Domain template path works
```bash
python run_v3.py data/sample-invoice.pdf \
  --extract \
  --schema_mode domain \
  --auto_schema \
  --output_dir output/v3
```

### Milestone C - Debug trace generation
```bash
python run_v3.py data/sample-invoice.pdf \
  --extract \
  --schema_mode domain \
  --auto_schema \
  --save_debug_traces \
  --output_dir output/v3
```

### Milestone D - Evaluate output quality
```bash
python run_v3.py data/sample-invoice.pdf \
  --extract \
  --schema_mode explicit \
  --schema_path schema_sample.json \
  --evaluate \
  --output_dir output/v3
```

---

## Definition of Done
1. `*_schema_discovery.json` and `*_target_schema.json` are both present and different by design.
2. `*_extraction_result.json` is schema-valid and non-empty for invoice + technical docs.
3. On failure, traces show parser and validation details.
4. Explicit runtime schema works for a previously unseen schema without code changes.

---

## Immediate Next Action
Start with Day 1 edits in `run_v3.py` and do Milestone A before touching model-level internals.
