# Document Intelligence Pipeline — Librarian v3

A **zero-shot, schema-conditioned document extraction pipeline** that accepts any PDF and any JSON schema at runtime, and produces structured output with full provenance, diagnostic traces, and measurable quality scores.

This project implements a **VLM Distillation framework** where a "Teacher" model (GPT-4o-V) was used to fine-tune a "Student" model (Llama-3.2-11B-Vision) for high-accuracy document hierarchy recognition.

> **Thesis system** — designed for reproducible, ablatable evaluation.  
> Powered by Hugging Face models: [RMunshi/vlm-student-thesis](https://huggingface.co/RMunshi/vlm-student-thesis) & [RMunshi/librarian-qwen-extractor](https://huggingface.co/RMunshi/librarian-qwen-extractor)

---

## Reproducibility

This is a thesis artifact. Reproducibility is treated as a first-class concern: every dependency, every model, every smoke-test document is pinned so that the numbers in the thesis can be regenerated from a clean checkout.

| Artifact | Role |
|---|---|
| `environment.yml` | Pinned Python + CUDA + ML-stack versions (single source of truth for deps). |
| `config/models.yaml` | HuggingFace asset manifest with revision SHAs — every model is fetched at the exact commit used in the thesis. |
| `scripts/bootstrap.py` | One-shot preflight + model download + smoke test, writes a `BOOTSTRAP_OK` sentinel summarising the run. |
| `data/simple_invoice.pdf`, `data/Super_Complex_2.pdf` | Committed smoke-test documents covering Invoice and Logistics domains. |

---

## Installation & Setup

**System Requirements**
- **OS:** Linux (Ubuntu 22.04 recommended)
- **GPU:** NVIDIA GPU, 24 GB VRAM (tested on RTX 3090, compute capability 8.6)
- **CUDA:** 12.4 host driver (CUDA toolkit ships via the pinned PyTorch wheel — no separate install needed)
- **Disk:** ~45 GB free at `$HF_HOME` (default `~/.cache/huggingface`) for the model cache
- **Conda:** Miniconda or Anaconda installed. If `conda` is not on your `PATH`, install Miniconda from <https://docs.conda.io/en/latest/miniconda.html> first.

**One-shot setup (three commands)**

```bash
git clone https://github.com/munshi007/document_intelligence_pipeline.git
cd document_intelligence_pipeline
conda env create -f environment.yml      # creates an isolated env named "silo"
conda activate silo
python scripts/bootstrap.py               # downloads pinned models + smoke tests
```

`bootstrap.py` is a one-time setup verifier. It will:
1. Verify Python, CUDA, and free disk.
2. Download every model in `config/models.yaml` at its pinned revision (~38 GB, shown with a confirmation prompt; pass `--yes` to skip).
3. Run the pipeline on both committed smoke PDFs and verify each produces non-empty extractions.
4. Write `BOOTSTRAP_OK` summarising the run.

If the script exits 0, the repo is reproducibly installed. After that, use `run_v3.py` directly for any document (see Quick Start below). For air-gapped environments, run bootstrap once online, then `export HF_HUB_OFFLINE=1` for subsequent runs.

> **Note:** If the conda env name `silo` collides with an existing env on your machine, override it with `conda env create -f environment.yml -n my_chosen_name` and adjust the `conda activate` line accordingly.

---

## Quick Start

After bootstrap succeeds, run the pipeline on any PDF. The system auto-discovers the document's domain, synthesises a schema, and emits structured JSON.

> **Bring your own documents:** drop any PDF into the `data/` directory (or pass an absolute path) — `data/` is gitignored except for the two committed smoke docs, so your own files stay local and never get committed by accident.

### Basic extraction (recommended)
```bash
python run_v3.py data/simple_invoice.pdf \
    --extract \
    --schema_mode auto \
    --output_dir output/my_run
```

### Following progress live
The pipeline logs to stderr. To follow progress and keep a transcript, redirect to a file you can `tail -f`:
```bash
python run_v3.py data/simple_invoice.pdf --extract --schema_mode auto \
    --output_dir output/my_run 2>&1 | tee output/my_run/run.log
# in another terminal:
tail -f output/my_run/run.log
```

### With debug traces
Saves every batch prompt, raw VLM output, and parse-failure log:
```bash
python run_v3.py data/simple_invoice.pdf \
    --extract \
    --schema_mode auto \
    --save_debug_traces \
    --output_dir output/debug_run
```

### With a fixed schema
If you need the output to conform to a strict pre-defined contract:
```bash
python run_v3.py data/simple_invoice.pdf \
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

## Project Structure

Where everything lives in the codebase:

```text
document_intelligence_pipeline/
├── run_v3.py                       # MAIN ENTRY POINT: pipeline orchestrator
├── environment.yml                 # Pinned conda env (reproducibility anchor)
├── pyproject.toml                  # Package metadata (no deps — see environment.yml)
├── config/
│   ├── pipeline_config.py          # Runtime thresholds, model defaults
│   └── models.yaml                 # HuggingFace asset manifest (pinned revision SHAs)
├── common/
│   ├── model_registry.py           # Reads models.yaml → serves (repo_id, revision) to loaders
│   └── vlm_providers/
│       ├── local_unsloth_provider.py # Llama VLM inference + JSON hallucination parsing
│       └── local_text_provider.py    # Qwen text extractor inference
├── core/
│   └── schemas.py                  # Pydantic base models for the Hierarchical Graph
├── converter/
│   └── engine.py                   # PDF rendering + vision model coordination
├── chunker/
│   └── graph_builder.py            # Physical layouts → Semantic Hierarchical Knowledge Graph
├── modules/
│   ├── layout_detector.py          # DocLayout-YOLO block-level detector
│   ├── table_structure_model.py    # Table Transformer (TATR)
│   └── layoutlm_classifier.py      # LayoutLMv3 token classifier
├── processors/
│   ├── reading_order.py            # AI-driven reading order logic
│   └── tables_v2/                  # Advanced table routing (Complex vs Simple tables)
├── extractor/
│   ├── discovery_agent.py          # Zero-shot domain scouting + schema synthesis
│   ├── agent.py                    # Batched Qwen extractor + retry path + verifier
│   ├── schema_engine.py            # Heuristic routing
│   └── evaluation.py               # Scorecard generation (incl. retry stats)
├── scripts/
│   ├── bootstrap.py                # One-shot reproducibility verifier (run after env create)
│   ├── eval_harness.py             # Multi-doc evaluation runner
│   ├── run_ablation.py             # 4-condition ablation study runner
│   └── generate_ground_truth.py    # Build GT annotations from your own PDFs
├── research/                       # Training-side code (the *trained models* live on HF, not here)
│   ├── distillation_agent.py       # Teacher → Student distillation orchestrator
│   ├── batch_distill.py            # Batched distillation runs over corpora
│   └── benchmarks/                 # Small-VLM comparison benchmarks
├── data/                           # Drop your own PDFs here (gitignored except the two smoke docs)
│   ├── simple_invoice.pdf          # Smoke-test doc 1 (Invoice domain)
│   └── Super_Complex_2.pdf         # Smoke-test doc 2 (Logistics domain)
└── output/                         # Created on first run (gitignored)
```

> **Note on training data:** The training corpora (`research/dataset/`), distilled outputs (`research/outputs/`), and the local model snapshots that were merged into the published HF checkpoints (`research/vlm_student_model/`, `research/librarian_qwen_specialist/`) are gitignored. The training scripts themselves ship so the pipeline is auditable end-to-end, but the actual fine-tuned model weights are distributed via Hugging Face (see `config/models.yaml`).

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

## Evaluation & Research Scripts

If you are replicating the thesis benchmarks. Note that `data/ground_truth/` is not shipped in this repo (the source PDFs come from third-party datasets and the annotations carry their licensing). To run these scripts you must either supply your own ground-truth JSONL or regenerate via `scripts/generate_ground_truth.py`.

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

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `conda: command not found` | Install Miniconda from <https://docs.conda.io/en/latest/miniconda.html> and re-open your shell. |
| `CondaValueError: prefix already exists` | An env named `silo` already exists. Use `conda env create -f environment.yml -n my_alt_name` and adjust `conda activate` accordingly. |
| Bootstrap fails at "CUDA not available" | Your NVIDIA driver isn't installed or visible. Verify with `nvidia-smi`. The pipeline can run on CPU but is unusably slow for the VLM. |
| Out-of-VRAM during smoke test | Models default to 4-bit quantisation but still need ~20 GB VRAM. A smaller GPU will OOM; cloud A10G / 3090 / 4090 / L40S all work. |
| `OSError: ... is gated` while downloading | Should not happen — all five models are public. If HF changes a model's visibility, run `huggingface-cli login` once. |
| Smoke test takes >10 minutes per PDF | Expected on first run (model warm-up). Subsequent runs use cached weights and are ~2× faster. |
| Air-gapped environment | Run `bootstrap.py` once on a machine with network, copy `~/.cache/huggingface` to the air-gapped host, then `export HF_HUB_OFFLINE=1`. |
