# Document Intelligence Pipeline — Librarian v3

A **zero-shot, schema-conditioned document extraction pipeline** that accepts any PDF and any JSON schema at runtime, and produces structured output with full provenance, diagnostic traces, and measurable quality scores.

This project implements a **VLM Distillation framework** where a "Teacher" model (GPT-4o-V) was used to fine-tune a "Student" model (Llama-3.2-11B-Vision) for high-accuracy document hierarchy recognition.

> **Thesis system** — designed for reproducible, ablatable evaluation.  
> Powered by Hugging Face models: [RMunshi/vlm-student-thesis](https://huggingface.co/RMunshi/vlm-student-thesis) & [RMunshi/librarian-qwen-extractor](https://huggingface.co/RMunshi/librarian-qwen-extractor)

---

## 🚀 Out-of-the-Box Hugging Face Integration

This repository is designed to run seamlessly **out-of-the-box**. You do **not** need to manually download massive model weights or configure complex model directories.

On your first run, the pipeline will automatically download and cache the necessary state-of-the-art models from the Hugging Face Hub:
1. **Llama-Vision (VLM-Student-Thesis):** For advanced semantic layout refinement.
2. **Qwen-Extractor (Librarian-Qwen):** For high-precision, JSON-structured technical extraction.
3. **DocLayout-YOLO:** For structural document layout parsing.
4. **Table Transformer (TATR):** For complex grid detection.

*Note: If you are running in an air-gapped environment, you can set `export HF_HUB_OFFLINE=1` to force the system to use locally cached snapshots.*

---

## 🛠️ Installation & Setup

**System Requirements:**
- **OS:** Linux (Ubuntu recommended)
- **GPU:** NVIDIA GPU with at least 16GB-24GB VRAM (e.g., RTX 3090 / 4090 / A10G)
- **CUDA:** 12.1 or 12.4 recommended

**1. Clone the repository:**
```bash
git clone https://github.com/munshi007/document_intelligence_pipeline.git
cd document_intelligence_pipeline
```

**2. Create a Conda Environment:**
```bash
conda create -n silo python=3.12 -y
conda activate silo
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```
*(Note: Unsloth and PyTorch may require specific installation commands depending on your CUDA version. See [Unsloth Installation](https://github.com/unslothai/unsloth) for details if needed).*

---

## 📖 Quick Start

You can run the pipeline on any PDF. The system will automatically route the document to the correct domain, synthesize a schema, and extract the structured data.

### Basic Extraction (Recommended)
This command automatically discovers the document's domain (e.g., Logistics, Hardware, Invoices) and generates a schema on the fly.
```bash
python run_v3.py data/sample.pdf \
    --extract \
    --schema_mode auto \
    --output_dir output/my_run
```

### Advanced Usage (Debugging & Custom Schemas)
Save detailed debug traces (prompts, raw VLM outputs) to see exactly how the AI is reasoning:
```bash
python run_v3.py data/sample.pdf \
    --extract \
    --schema_mode auto \
    --save_debug_traces \
    --output_dir output/debug_run
```

**Force a Custom JSON Schema:**
If you have a strict data contract you need the document to conform to:
```bash
python run_v3.py data/sample.pdf \
    --extract \
    --schema_mode explicit \
    --schema_path my_custom_schema.json \
    --output_dir output/explicit_run
```

---

## 📊 Output Artifacts

All results are saved in the directory specified by `--output_dir` (default is `output/v3/`).

| File | Description |
|------|-------------|
| `<doc>_universal_extraction.json` | Final extracted structured JSON payload. This is the main output. |
| `<doc>_auto_schema.json` | The dynamically generated JSON schema contract used for extraction. |
| `<doc>_discovery.json` | The AI Agent's domain routing decision + confidence score. |
| `<doc>_graph_summary.txt` | Human-readable breakdown of the Hierarchical Knowledge Graph. |
| `<doc>_manifest.json` | Page-by-page breakdown of detected layout blocks (images, text, tables). |
| `debug_traces/` | *(If enabled)* Per-batch raw outputs, VLM prompts, and parse failure logs. |

---

## 🏗️ Project Structure Explained

Here is where everything lives in the codebase so you can easily navigate and extend the system:

```text
document_intelligence_pipeline_original/
├── run_v3.py                       # MAIN ENTRY POINT: The pipeline orchestrator
├── config/                         
│   └── pipeline_config.py          # Central configuration (thresholds, model defaults)
├── core/                           
│   └── schemas.py                  # Pydantic base models for the Hierarchical Graph
├── chunker/
│   └── graph_builder.py            # Converts physical layouts → Semantic Hierarchical Knowledge Graph
├── converter/
│   └── engine.py                   # Handles PDF rendering and coordinates the Vision models
├── modules/
│   ├── layout_detector.py          # DocLayout-YOLO implementation
│   └── table_structure_model.py    # TATR (Table Transformer) implementation
├── processors/                     
│   ├── reading_order_planner.py    # AI-driven reading order logic
│   └── tables_v2/                  # Advanced table routing (Complex vs Simple tables)
├── extractor/                      
│   ├── discovery_agent.py          # Zero-shot document scouting (Schema Generation)
│   ├── agent.py                    # Batched Qwen extractor + additive synthesis
│   ├── schema_engine.py            # Heuristic routing
│   └── evaluation.py               # Evaluation metrics generation
├── common/
│   └── vlm_providers/
│       └── local_unsloth_provider.py # Unsloth inference layer + JSON hallucination parsing
└── scripts/
    ├── eval_harness.py             # Multi-doc evaluation runner for benchmarks
    └── run_ablation.py             # Script for testing 4-condition ablation studies
```

---

## 📈 Benchmarks & Performance

The Librarian v3 architecture was evaluated against a ground-truth dataset of highly complex, multi-page technical datasheets and logistics invoices.

| Architecture | Model | Schema Validity | Required Field Completion | F1 Score (Complex Tables) | Routing Accuracy |
|--------------|-------|-----------------|---------------------------|---------------------------|------------------|
| Baseline VLM | GPT-4o | 82.5% | 78.1% | 61.2% | N/A (Zero-shot) |
| Standard OCR | LayoutLMv3 | 65.0% | 54.3% | 42.8% | 81.0% |
| **Librarian v3** | **vlm-student-thesis + qwen-extractor** | **98.2%** | **96.5%** | **92.4%** | **97.8%** |

### Key Advantages:
1. **Additive Synthesis:** By breaking documents into semantic batches (Graph Nodes), the Librarian v3 architecture overcomes the "lost in the middle" hallucination problem typical in standard LLM document extraction.
2. **Schema-Adherence:** The custom Qwen extractor is fine-tuned to never hallucinate wrapper keys, ensuring a 98.2% validity rate for strict JSON schemas.
3. **Complex Grid Solving:** The table router detects multi-span/ruled tables and shifts processing from text-based extraction to spatial TATR extraction seamlessly.

---

## 🧬 Model Weights & Training

The fine-tuned Vision-Language Model (VLM) weights produced by this pipeline are hosted on Hugging Face:

👉 **[RMunshi/vlm-student-thesis](https://huggingface.co/RMunshi/vlm-student-thesis)**

### Training Metrics
Training was completed with a final loss of **~0.08** over 1000 steps. Full metrics, loss curves, and hardware usage are documented on the [Weights & Biases Dashboard](https://wandb.ai/rohanmunshi06-otto-von-guericke-university-magdeburg/huggingface/runs/zxxsiwz3).

---

## 🔬 Evaluation & Research Scripts

If you are replicating the thesis benchmarks:

**Multi-Document Evaluation:**
```bash
python scripts/eval_harness.py \
    --pdf_dir /path/to/pdfs \
    --n_docs 30 \
    --schema_mode domain \
    --ground_truth data/ground_truth/annotations.jsonl \
    --output_dir output/eval_run
```

**Distillation Effectiveness (Specialist vs Teacher):**
```bash
python scripts/eval_specialist_vs_teacher.py \
    --pdf_dir /path/to/pdfs \
    --n_docs 30 \
    --specialist_model RMunshi/librarian-qwen-extractor \
    --teacher_model gpt-4o \
    --ground_truth data/ground_truth/annotations.jsonl \
    --output_dir output/distillation_eval
```
