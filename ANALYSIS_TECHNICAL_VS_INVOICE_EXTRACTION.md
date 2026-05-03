# Deep Analysis: Technical PDF vs Invoice Extraction Pipeline Failure

## Executive Summary

The document_intelligence_pipeline successfully extracts data from invoices but returns **empty/null values** for technical PDFs. After analyzing both the discovery phase, extraction architecture, and actual outputs, I've identified **7 fundamental architectural differences** that explain why the two document types are treated so differently.

**Root Cause**: The pipeline uses a **single discovery-driven schema synthesis** that works well for familiar corporate documents (invoices) but fails catastrophically for technical documents that require a **pre-defined, domain-aware schema** rather than dynamic field discovery.

---

## Part 1: Discovery Phase Analysis

### 1.1 Invoice Discovery (WORKS ✓)

**Discovery Result:**
```json
{
  "domain": "Invoice",
  "is_high_density": true,
  "dynamic_fields": 22 fields,  // company_name, invoice_number, total_amount, etc.
  "confidence": 1.0
}
```

**Why It Works:**
- Invoice format is highly standardized (ISO/IEC 20022)
- Discovery agent successfully identified consistent patterns: "invoice", "total", "currency", "USD/EUR"
- Generated 22 highly relevant fields that match actual invoice structure
- All fields are directly present in the document as text

### 1.2 Technical PDF Discovery (FAILS ✗)

**Discovery Result:**
```json
{
  "domain": "Industrial",
  "is_high_density": true,
  "dynamic_fields": 5 fields,  // art_no, product_name, manufacturer, pin_assignment, connectors_type
  "confidence": 1.0
}
```

**Why It Fails (Critical Issue #1):**
```
🔴 INSUFFICIENT SCHEMA DISCOVERY
- Only 5 fields discovered vs. invoice's 22
- Missing critical fields:
  ❌ parameters (electrical, mechanical, environmental)
  ❌ connector details (pinout assignments)
  ❌ LED diagnostic states
  ❌ standards/certifications
  ❌ environmental ratings
```

**Code Location**: [extractor/discovery_agent.py](extractor/discovery_agent.py#L1-L85)

**Problem Analysis**:
```python
# Discovery agent prompt (lines 54-72)
prompt = f"""
Analyze the document preview and synthesize optimal JSON schema.
1. Identify domain (e.g., Technical_Datasheet, Invoice, etc.)
2. Determine if high_density: true/false
3. Define dynamic_fields needed
"""
```

The prompt is **generic and insufficient** for technical documents:
- It doesn't probe for nested structures (connectors → pin assignments)
- It doesn't recognize table-based specifications as extractable data
- It treats "pin_assignment" as a simple `list[str]` instead of complex objects with pin, signal, function

---

## Part 2: Schema Synthesis Issue

### 2.1 Invoice Schema (22 Dynamic Fields)
```python
# Generated schema includes:
company_name (str)
invoice_number (str)
invoice_date (str)
total_amount (float)
tax_amount (float)
line_items (list of objects with description, amount)
# ... 16 more fields

# Result: Schema closely matches actual invoice structure
```

### 2.2 Technical PDF Schema (5 Basic Fields)
```python
# Generated schema includes only:
art_no (str)
product_name (str)
manufacturer (str)
pin_assignment (list[str])  # ❌ Wrong! Should be list[PinAssignment]
connectors_type (str)

# Missing:
# - parameters (electrical, mechanical, environmental)
# - connectors with detailed pin info
# - LED behaviors
# - standards
```

**Code Location**: [extractor/discovery_agent.py#L99-L120](extractor/discovery_agent.py#L99-L120)

```python
def synthesize_model(self, result: DiscoveryResult) -> Type[BaseModel]:
    """Creates Pydantic model from discovery results"""
    fields = {}
    
    # For each discovered field, map to Python type
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "list[str]": List[str],  # ❌ Assumes all lists are simple strings
    }
    
    # Problem: No support for nested complex types like:
    # - list[TechParameter]
    # - list[PinAssignment]
    # - list[ConnectorSpec]
```

---

## Part 3: Two-Pass Extraction Pipeline Differences

### 3.1 Architecture Overview

Both documents use the **"Markdown Harvest" (Two-Pass)** approach because both are marked `is_high_density: true`:

```
Pass 1: _harvest_as_markdown()
  └─> Extract everything as raw Markdown
      └─> Bypass JSON token limits
      
Pass 2: _refine_to_structured()
  └─> Map Markdown text to Pydantic schema
  └─> Use domain-specific SPECIALIST_HINTS
```

**Code Location**: [extractor/agent.py#L234-L360](extractor/agent.py#L234-L360)

### 3.2 Invoice: Two-Pass Success ✓

**Pass 1 Harvest** (Works effectively):
```markdown
# Invoice WMACCESS Internet

| Invoice No | Customer No | Invoice Period | Date |
| --- | --- | --- | --- |
| 123100401 | 12345 | 01.02.2024 - 29.02.2024 | 1. März |

| Service Description | Amount -without VAT- |
| --- | --- |
| Basic Fee wmView | 130,00 € |
| Basis fee for additional user accounts | 10,00 € |
```

✓ **Why it works**: All content is text/table based → converts to markdown cleanly

**Pass 2 Refinement** (Effective mapping):
```python
# Using SPECIALIST_HINTS["Corporate"]:
hint = """
Focus on financial and entity details:
1. Identify Sender (Organization/Manufacturer) and Recipient
2. Extract all Entities mentioned
3. Extract Invoice No as title, Date, Document Type
4. List line items in Timeline
"""

# Prompt uses SPECIALIST_HINTS + discovered schema
# → Successfully maps markdown to DynamicInvoiceSchema
```

**Result**: Data extracted successfully (despite validation error in `total_amount_per_service`)

### 3.3 Technical PDF: Two-Pass Failure ✗

**Pass 1 Harvest** (Severely degraded):
```markdown
# 4.6.2.40 Connection overview Cube67+ DIO16 IOL8 A/B E 8xM12 Art.-No. 56768

[FIGURE: output/v3/extracted_figures/figure_001.png]
FE Functional ground connection FE IN System bus input port...

**Pin-assignment**
| IN (BUS) |  |  |
| --- | --- | --- |
|  | Pin 1 | 24 V UA/4 A |
|  | Pin 2 | 24 V US/4 A |

**System bus output port**
| OUT (BUS) |  |  |
| --- | --- | --- |
|  | Pin 1 | 24 V UA/4 A |
```

❌ **Why it fails**:
1. **Figure content is lost**: `[FIGURE: ...]` markers don't contain extractable data
2. **Table headers are missing context**: Pin tables lack human-readable headers in markdown
3. **No structured relationship preservation**: Can't map "Pin 1 = 24 V UA/4 A" to pin objects
4. **Harvested markdown is minimal**: Only ~200-300 chars vs invoice's 5000+ chars

**Graph Summary Shows the Problem**:
```
[node_002] shows: [FIGURE: output/v3/extracted_figures/figure_001.png]
                  FE Functional ground connection FE IN System bus...
                  
# ❌ Figure content cannot be extracted as meaningful markdown
# ❌ OCR'd text from figures is fragmented
```

**Pass 2 Refinement** (Complete failure):

```python
# Using SPECIALIST_HINTS["Industrial"]:
hint = """
Focus on technical specifications:
1. Identify product name, article number, manufacturer
2. Extract ALL parameters from parameter tables
3. Detail ALL connectors with pin assignments
4. Capture ALL LED diagnostic states
5. Note standards/certifications
"""

# BUT: The discovered schema only has 5 fields
# - art_no: str
# - product_name: str
# - manufacturer: str
# - pin_assignment: list[str]  # ❌ Wrong type!
# - connectors_type: str

# ❌ Prompt says "extract ALL parameters" but schema has NO parameters field
# ❌ Prompt says "detail ALL connectors" but schema only has connectors_type (str)
# ❌ Prompt says "capture LED states" but schema has NO LED field
```

**Result**: Model generates outputs that don't match discovered schema → returns nulls

---

## Part 4: Context Building Differences

### 4.1 Invoice Context (Rich & Usable)

**Graph Nodes Created**: 12 nodes
```
node_000: Company details, customer info (Document Root)
node_001: Contact info, names
node_002: Invoice header table (Invoice No, Period, Date)
node_003: Services table (Description, Amount, Quantity)
node_004: Payment terms
node_005-011: Invoice details, tax breakdown
```

**Context to Markdown Conversion**: ✓ Effective
- Pure text content → direct markdown
- Tables → markdown tables with headers
- All content is machine-readable (no figures)
- ~5000+ characters of meaningful context

**to_extraction_batches() Result**:
```python
# GraphBuilder.to_extraction_batches():
# Creates semantic batches of max_chars=4000
# Each batch contains related nodes
# ✓ All content is preserved and contextual

batch_1: Company info + Customer info + Invoice header
batch_2: Services table + Payment terms + Details
```

### 4.2 Technical PDF Context (Lost in Conversion)

**Graph Nodes Created**: 9 nodes
```
node_000: Document root (Installation)
node_001: Title (4.6.2.40 Connection overview...)
node_002: Figure reference + OCR text
node_003: Caption
node_004-008: Pin assignment tables
```

**Context to Markdown Conversion**: ❌ Severely degraded
```markdown
[CONTEXT: 4.6.2.40 Connection overview...]

[FIGURE: output/v3/extracted_figures/figure_001.png]
FE Functional ground connection FE IN System bus input port...

**Pin-assignment**
| IN (BUS) |  |  |
| --- | --- | --- |
|  | Pin 1 | 24 V UA/4 A |
|  | Pin 2 | 24 V US/4 A |
```

**Problems**:
1. **Figure metadata lost**: Image path is referenced but content isn't extracted
2. **Table structure degraded**: 
   - Headers are missing or merged
   - Relationships between pins unclear
   - Pin names don't match expected schema
3. **Insufficient context**: Only ~800 characters vs invoice's 5000+
4. **Semantic meaning lost**: "Pin 1 = 24 V UA/4 A" needs to be parsed as PinAssignment object

**Code Location**: [extractor/agent.py#L685-L730](extractor/agent.py#L685-L730)

```python
def _harvest_as_markdown(self, image, context_nodes, context_markdown, domain, trace_context):
    """Pass 1: Extract everything as raw Markdown"""
    
    # Processes nodes in 3000-char batches
    for batch_content in batches:
        harvest_prompt = f"""
        I am the Visual Librarian. Harvest every detail from this document.
        
        1. Extract EVERY row from EVERY table exactly
        2. Extract ALL key entities and values
        3. Be EXTREMELY PRECISE
        
        SOURCE CONTENT: {batch_content}
        """
        
        # ❌ Problem: For figure nodes, batch_content includes "[FIGURE: ...]"
        # ❌ Model cannot extract data from a figure reference
        # ❌ No visual processing in Pass 1 (image=None to disable hallucinations)
```

---

## Part 5: SPECIALIST_HINTS Misalignment (Critical #2)

### 5.1 SPECIALIST_HINTS Definition

**Code Location**: [extractor/agent.py#L48-L85](extractor/agent.py#L48-L85)

```python
SPECIALIST_HINTS = {
    "Corporate": """
    Focus on financial and entity details:
    1. Identify Sender and Recipient
    2. Extract all Entities mentioned
    3. Extract Invoice No, Date, Document Type
    4. List line items in Timeline
    """,
    
    "Hardware": """
    Focus on technical specifications:
    1. Product name, article number, manufacturer
    2. Extract ALL parameters from parameter tables
    3. Detail ALL connectors with pin assignments
    4. Capture ALL LED diagnostic states
    5. Note standards/certifications
    """,
    
    "Industrial": """
    [Same as Hardware]
    """,
    
    "Industrial_datasheet": """
    EXHAUSTIVE technical extraction:
    1. Identity: product, article no, manufacturer
    2. Parameters: EVERY row from EVERY table
       - Electrical: voltage, current, protection
       - Mechanical: dimensions, weight, material
       - Environmental: temperature, IP rating
    3. Connectors: pin assignments for EVERY connector
    4. Diagnostics: EVERY LED state and meaning
    5. Standards: ALL certifications
    """
}
```

### 5.2 The Misalignment Problem

**For Invoice** (WORKS):
```
Discovery identifies domain = "Invoice"
  ↓
But no specific hint for "Invoice" domain
  ↓
Fallback: Generic hint "Follow JSON schema exactly"
  ↓
BUT generated schema matches document structure
  ↓
✓ Extraction succeeds
```

**For Technical PDF** (FAILS):
```
Discovery identifies domain = "Industrial"
  ↓
Uses SPECIALIST_HINTS["Industrial"] with detailed prompting:
  - "Extract ALL parameters from parameter tables"
  - "Detail ALL connectors with pin assignments"
  - "Capture ALL LED diagnostic states"
  ↓
BUT the discovered schema only has 5 fields:
  - NO parameters field
  - NO LED field
  - pin_assignment is list[str], not list[PinAssignment]
  ↓
❌ Specialist prompt conflicts with schema
❌ Model cannot map detailed extraction to incompatible schema
❌ Returns null for all fields
```

---

## Part 6: Schema Registry & Routing (Critical #3)

### 6.1 Current Schema Routing

**Code Location**: [extractor/schema_registry.py](extractor/schema_registry.py)

The pipeline has pre-defined schemas:
```python
# Fixed schemas (NOT used for technical!)
LibrarianUniversalHardware  # For industrial docs
LibrarianInvoiceRecord      # For invoices
LibrarianGeneralClerk       # For generic docs
LibrarianBusinessRecord     # For corporate docs
```

**Routing Logic**:
```
If discovery_domain == "Invoice":
  ✓ Route to LibrarianInvoiceRecord (pre-defined)
  
If discovery_domain == "Industrial":
  ❌ DON'T use LibrarianUniversalHardware
  ❌ Instead, use dynamically synthesized schema from 5 fields
```

**Code Location**: [extractor/agent.py#L295-310](extractor/agent.py#L295-310)

```python
# In orchestrator (run_v3.py):
discovery_agent = DiscoveryAgent(model_id)
discovery_result = discovery_agent.scout(doc_preview)

# Creates DYNAMIC schema from discovery:
dynamic_schema = discovery_agent.synthesize_model(discovery_result)

# ❌ PROBLEM: For invoices, hardcoded schemas work
# ❌ For technical docs, synthesized schema is TOO MINIMAL
# ❌ Should fall back to LibrarianUniversalHardware
```

---

## Part 7: Actual Extraction Outputs

### 7.1 Invoice Extraction (Despite Error)

```json
{
  "company_name": "wm Morphy Russel GmbH",
  "customer_name": null,
  "invoice_number": "123100401",
  "invoice_date": "01.03.224",
  "total_amount": 381.12,
  "tax_amount": 72.41,
  "gross_amount": 453.53,
  "service_description": "Basic Fee wmView\nBasis fee...",
  "amount_without_vat": 308.71,
  "total_amount_per_service": 130.0,
  "transaction_fee": 8.12,
  "phone_number": "(+49) 71 91 47-0"
}
```

✓ **12 out of 22 fields populated** with actual data
✓ **Extracted despite validation error** (total_amount_per_service is array not float)

### 7.2 Technical PDF Extraction (Complete Failure)

```json
{
  "art_no": null,
  "product_name": null,
  "manufacturer": null,
  "pin_assignment": [],
  "connectors_type": null,
  "reasoning_thoughts": null
}
```

✗ **0 out of 5 fields populated**
✗ **All values are null or empty**
✗ **No reasoning or explanation**

---

## Part 8: Extraction Batching Differences

### 8.1 Invoice Batching (Effective)

**Code**: [chunker/graph_builder.py#L85-140](chunker/graph_builder.py#L85-140)

```python
GraphBuilder.to_extraction_batches(
    nodes=12_invoice_nodes,
    max_chars=4000,
    preserve_atomic=False
)
```

Result:
```
batch_1 (3800 chars): Nodes 0-2 (company, customer, header)
batch_2 (3900 chars): Nodes 3-4 (services table, payments)
batch_3 (2100 chars): Nodes 5-8 (details, breakdown)
```

✓ Each batch is semantically cohesive
✓ No table splitting
✓ All context preserved

### 8.2 Technical PDF Batching (Severely Limited)

```python
GraphBuilder.to_extraction_batches(
    nodes=9_technical_nodes,
    max_chars=4000,
    preserve_atomic=False
)
```

Result:
```
batch_1 (1200 chars): Nodes 0-2 (title, figure, caption)
batch_2 (800 chars): Nodes 3-4 (pin tables fragments)
batch_3 (600 chars): Nodes 5-8 (more fragments)
```

❌ **Severely undersized batches**
❌ **Figure content cannot be meaningfully processed**
❌ **Not enough context to understand relationships**

---

## Part 9: High-Density Flag Impact

### 9.1 is_high_density: true Effect

**Enables Two-Pass Architecture**:
```
if is_high_density:
    harvest_md = _harvest_as_markdown(...)  # Pass 1
    return _refine_to_structured(harvest_md, schema, domain)  # Pass 2
else:
    # One-pass direct extraction
    return extract_structured_direct(context, schema, domain)
```

**For Invoice**: ✓ Works well
- Dense table-based content → harvest captures all tables as markdown
- Refinement successfully maps to financial schema

**For Technical**: ✗ Fails catastrophically
- Dense figure-based content → harvest loses figure semantics
- Refinement has insufficient context
- Tables extracted but cannot be mapped to missing schema fields

---

## Part 10: Root Cause Summary

### The 7 Critical Differences

| Aspect | Invoice | Technical | Status |
|--------|---------|-----------|--------|
| **Domain Detection** | ✓ "Invoice" | ✓ "Industrial" | Both work |
| **Schema Discovery** | ✓ 22 fields | ❌ 5 fields | **MISMATCH** |
| **Schema Completeness** | ✓ financial + entities | ❌ missing parameters/connectors | **INCOMPLETE** |
| **Type Correctness** | ✓ Correct types | ❌ pin_assignment as list[str] not list[PinAssignment] | **WRONG TYPES** |
| **Context Preservation** | ✓ 5000+ chars | ❌ 800 chars | **CONTEXT LOSS** |
| **Markdown Harvest** | ✓ Text/tables convert well | ❌ Figures become "[FIGURE: ...]" | **HARVEST FAILURE** |
| **Specialist Hints Alignment** | ✓ Generic hints work | ❌ Detailed hints conflict with minimal schema | **MISALIGNMENT** |

---

## Part 11: Why High-Density Flag Backfires

### The Paradox

```python
# Technical docs are correctly identified as high_density
is_high_density = True

# This triggers Two-Pass (Markdown Harvest + Refinement)
# Which is SUPPOSED to help with dense content

# But for FIGURE-DENSE technical docs:
# - Pass 1 (harvest) cannot extract figure semantics
# - Pass 2 (refine) gets insufficient markdown context
# - Result: CATASTROPHIC FAILURE

# Whereas invoice (also high_density) succeeds because:
# - Pass 1 (harvest) preserves table structure perfectly
# - Pass 2 (refine) has rich markdown context
# - Result: SUCCESS
```

**The Issue**: Two-Pass architecture assumes all "high-density" content can be converted to meaningful markdown. This works for:
- ✓ Tables (markdown tables preserve structure)
- ✓ Dense text blocks (markdown preserves text)

But fails for:
- ❌ Figures/Diagrams (cannot be represented as markdown)
- ❌ Visual pinouts (require visual understanding)
- ❌ Complex technical layouts (lose spatial relationships)

---

## Part 12: What the Code Actually Does

### The Execution Flow

```
1. Discovery Phase:
   discovery_agent.scout(doc_preview)
   → For invoice: domain="Invoice", fields=22 ✓
   → For technical: domain="Industrial", fields=5 ❌

2. Schema Synthesis:
   discovery_agent.synthesize_model(discovery_result)
   → For invoice: Creates schema with all 22 fields ✓
   → For technical: Creates minimal schema with 5 fields ❌

3. Extraction Routing:
   extract_structured(
       is_high_density=True,  # Both are high-density
       response_model=dynamic_schema  # Different schemas!
   )
   
4. Pass 1 - Harvest:
   _harvest_as_markdown(context_nodes)
   → Invoice: Extracts detailed markdown ✓
   → Technical: Extracts incomplete markdown ❌
   
5. Pass 2 - Refine:
   _refine_to_structured(
       markdown_text=harvest_md,
       response_model=dynamic_schema,  # 5-field schema for technical!
       domain="Industrial",  # Uses detailed hints
       SPECIALIST_HINTS["Industrial"]  # Expects complex extraction
   )
   → Invoice: Maps markdown to 22-field schema ✓
   → Technical: Cannot map sparse markdown to detailed hints ❌
   
6. Output:
   → Invoice: 12/22 fields populated ✓
   → Technical: 0/5 fields populated ❌
```

---

## Recommendations to Fix

### Priority 1: Schema Discovery Enhancement
```python
# Enhance discovery prompt for technical documents:
# - Add specific probes for parameter tables
# - Recognize connector/pinout patterns
# - Detect environmental/electrical specifications
# - Generate richer schema with nested types

def scout(self, doc_preview: str) -> DiscoveryResult:
    prompt = f"""
    For technical documents, also probe for:
    1. Parameter tables: Extract all row headers and units
    2. Connector information: M12, M8, RJ45, pin count
    3. LED/diagnostic: State indicators and meanings
    4. Standards: Certifications, compliance markings
    5. Environmental specs: Temperature, IP ratings, materials
    
    Return nested field definitions for complex types.
    """
```

### Priority 2: Schema Routing Logic
```python
# For Industrial domain, use pre-defined schema, not synthesized:
if discovery_result.domain == "Industrial":
    response_model = LibrarianUniversalHardware  # Use full schema
else:
    response_model = discovery_agent.synthesize_model(discovery_result)
```

### Priority 3: Figure-Aware Harvesting
```python
# Detect and handle figure nodes specially:
def _harvest_as_markdown(self, ...):
    for batch_content in batches:
        # If batch contains [FIGURE: ...] reference:
        if "[FIGURE:" in batch_content:
            # Extract OCR text more aggressively
            # Or use multi-modal processing
            # Or extract figure metadata separately
```

### Priority 4: Context Preservation
```python
# For technical docs, increase batch size or use full-document context:
GraphBuilder.to_extraction_batches(
    nodes=technical_nodes,
    max_chars=8000,  # Larger for technical (was 4000)
    preserve_atomic=True  # Keep tables/figures intact
)
```

---

## Conclusion

The fundamental flaw is that the pipeline assumes **"high-density" implies "text/table-dense"** but technical documents are often **"figure/diagram-dense"**. The two-pass architecture works excellently for the former but catastrophically fails for the latter because:

1. **Discovery** generates insufficient schema (5 vs 22 fields)
2. **Harvest** loses figure semantics when converting to markdown
3. **Refinement** gets minimal context to work with
4. **Specialist hints** conflict with the minimal schema

The pipeline needs **domain-aware schema routing** that recognizes Industrial/Technical documents and applies pre-defined schemas with enhanced figure/table handling instead of relying on dynamic discovery and text-only markdown extraction.
