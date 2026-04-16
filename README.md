# Document Intelligence Pipeline (VLM Distillation)

A professional, modular PDF processing pipeline for layout detection, table extraction, and visual structural analysis. 

This project is the implementation of a VLM Distillation framework where a "Teacher" model (GPT-4o-V) was used to fine-tune a "Student" model (Llama-3.2-11B-Vision) for high-accuracy document hierarchy recognition.

## Model Weights

The fine-tuned Vision-Language Model (VLM) weights produced by this pipeline are hosted on Hugging Face:

👉 **[RMunshi/vlm-student-thesis](https://huggingface.co/RMunshi/vlm-student-thesis)**

The model supports structured JSON output for document layout analysis, identifying:
- Visual Hierarchy (Titles, Headers H1-H3)
- Typography (Font size, Bold/Italic)
- Structural reasoning

## 📂 Project Structure

```
.
├── research/          # VLM Distillation scripts & training code
│   ├── train_student.py    # Main fine-tuning script
│   └── test_model_output.py # Inference & validation
├── pipeline/          # Core processing orchestration
├── models/            # AI model wrappers (OCR, Layout, VLM)
├── processors/        # Specialized document logic (Tables, Reading Order)
├── config/            # Pipeline configuration
├── common/            # Shared types and utilities
└── cli.py             # Command-line interface
```

## 🚀 Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   python run_pipeline.py --input data/your_pdf.pdf
   ```

## 📊 Training Metrics

Training was completed with a final loss of **~0.08** over 1000 steps. Full metrics, loss curves, and hardware usage are documented on the [Weights & Biases Dashboard](https://wandb.ai/rohanmunshi06-otto-von-guericke-university-magdeburg/huggingface/runs/zxxsiwz3).

---
**Main Branch**: `main` (Please use this branch for the most updated code).
