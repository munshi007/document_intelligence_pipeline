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
| `data/PDFS/*.pdf` | Ten committed smoke-test / regression PDFs (invoices, industrial datasheets, complex tables). Bootstrap smoke-tests `simple_invoice.pdf` (Invoice) and `Super_Complex_2.pdf` (Industrial). |

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

> **Bring your own documents:** drop any PDF into `data/` (or pass an absolute path). The committed test set lives in `data/PDFS/`; everything else under `data/` is gitignored, so your own files stay local and never get committed by accident.

### Basic extraction (recommended)
```bash
python run_v3.py data/PDFS/simple_invoice.pdf \
    --extract \
    --schema_mode auto \
    --output_dir output/my_run
```

### Following progress live
The pipeline logs to stderr. To follow progress and keep a transcript, redirect to a file you can `tail -f`:
```bash
python run_v3.py data/PDFS/simple_invoice.pdf --extract --schema_mode auto \
    --output_dir output/my_run 2>&1 | tee output/my_run/run.log
# in another terminal:
tail -f output/my_run/run.log
```

### With debug traces
Saves every batch prompt, raw VLM output, and parse-failure log:
```bash
python run_v3.py data/PDFS/simple_invoice.pdf \
    --extract \
    --schema_mode auto \
    --save_debug_traces \
    --output_dir output/debug_run
```

### With a fixed schema
If you need the output to conform to a strict pre-defined contract:
```bash
python run_v3.py data/PDFS/simple_invoice.pdf \
    --extract \
    --schema_mode explicit \
    --schema_path my_custom_schema.json \
    --output_dir output/explicit_run
```

### With hint fields (force-include specific fields)
When you want auto-discovery **plus** a guarantee that certain fields appear in the output — populated when the document supplies them, `null` when it doesn't:
```bash
python run_v3.py data/PDFS/simple_invoice.pdf \
    --extract \
    --schema_mode auto \
    --hint_fields "vendor_tax_id,warranty_period,part_number" \
    --output_dir output/hint_run
```

Each hinted field is injected into the discovered schema as `Optional[str]`. The LLM is told the field exists and tries to fill it from the document; if it's truly absent, the output value is `null`. Use this when you want a stable output contract without writing a full JSON schema — for complete control over types and nested structures, use `--schema_mode explicit --schema_path …` instead.

---

## Modular CLI — run one stage at a time

`run_v3.py` above runs the whole pipeline in one shot. The same pipeline is also exposed as a **per-stage CLI**, where each stage is its own subcommand — useful for debugging, re-running a single stage, or feeding a downstream tool the intermediate artifacts:

```bash
python -m cli run-all data/PDFS/simple_invoice.pdf --extract   # full pipeline
python -m cli pdf-to-layout data/PDFS/simple_invoice.pdf        # just one stage
```

(`python cli.py <subcommand>` is equivalent.)

| Subcommand | Stage | Loads |
|---|---|---|
| `pdf-to-layout` | 1 · detect layout regions | Vision model |
| `pdf-to-markdown` | 2 · structured Markdown + manifest | Vision model |
| `md-to-graph` | 3 · Hierarchical Knowledge Graph | Vision model |
| `discover-schema` | 4 · synthesise the extraction schema | Text model |
| `extract` | 5 · extract structured JSON | Text model |
| `pdf-to-graph` | *composite:* stages 1–3 in one process | Vision model |
| `discover-and-extract` | *composite:* stages 4–5 in one process | Text model |
| `run-all` | every stage, end-to-end | both (isolated) |

**Two behaviours make the stages composable:**

- **Smart-skip** — a stage reuses existing output instead of recomputing it. Pass `--force` to recompute.
- **Auto-chain** — call any stage and its missing prerequisites run automatically. e.g. `extract` on a fresh PDF transparently runs layout → markdown → graph → discovery first.

**GPU isolation:** `run-all` chains the work as two subprocess groups — `pdf-to-graph` (Vision) then `discover-and-extract` (Text) — so the two models never share VRAM in the same Python process. Those composite subcommands are exactly those groups, and are runnable on their own.

Common flags: `--output-dir/-o`, `--schema-mode auto|explicit`, `--schema-path`, `--hint-fields`, `--with-grounding`, `--save-debug-traces`, `--max-pages`, `--debug`, `--force`.

`--hint-fields` accepts a comma-separated list (e.g. `--hint-fields "vendor,warranty,part_number"`) and forces those fields into the discovered schema as `Optional[str]` — they'll appear in the extraction output, populated if found in the document or `null` if absent. Honored at the discovery stage; if discovery artifacts already exist on disk you'll be warned to pass `--force` to re-discover.

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
├── run_v3.py                       # One-shot entry point (monolithic orchestrator)
├── cli.py                          # Per-stage CLI (`python -m cli <stage>`)
├── stages/                         # One module per pipeline stage
│   ├── layout.py, markdown.py, graph.py   # Vision-model stages (1–3)
│   ├── discovery.py, extract.py           # Text-model stages (4–5)
│   ├── orchestrate.py              # In-process stage-group runners (Vision / Text)
│   └── serialize.py, paths.py      # Shared serialisation + canonical output paths
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
├── data/                           # Drop your own PDFs here (gitignored except data/PDFS/)
│   └── PDFS/                       # Committed test set — 10 PDFs, clone-and-run
│       ├── simple_invoice.pdf      # Smoke-test doc 1 (Invoice domain)
│       └── Super_Complex_2.pdf     # Smoke-test doc 2 (Industrial datasheet)
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
