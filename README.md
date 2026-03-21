# PDF Pipeline

A professional, modular PDF processing pipeline for layout detection, table extraction, and text recovery.

## Project Structure

```
pdf_pipeline/
├── config/         # Configuration modules
├── models/         # AI model wrappers
├── processors/     # Processing logic
├── analysis/       # Document analysis
├── output/         # Output generation
├── utils/          # Shared utilities
├── scripts/        # Diagnostic tools
├── data/           # Input PDFs
└── weights/        # Model weights
```

## Usage

```bash
python cli.py data/sample.pdf --output results/
```

## Requirements

See `requirements.txt`.
