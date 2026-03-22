"""
Configuration settings for the Enhanced PDF Processing Pipeline
"""

import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    'confidence_threshold': 0.20,  # Lowered to 0.20 to catch missing regions (VLM will filter)
    'layout_detection_dpi': 300,
    'thumbnail_dpi': 150,
    'table_detection_threshold': 0.5, # Lowered to 0.5
}

# Centralized Weights Paths (pointing to the clean 'models/' directory)
WEIGHTS_CONFIG = {
    'layout_config': 'models/layout/publaynet_config.yaml',
    'layout_weights': 'models/layout/publaynet_weights.pkl',
    'custom_model': 'models/custom/',
    'table_model': 'models/table/',
    'ocr_model': 'models/ocr/',
}

# VLM Configuration (SOTA Agentic Layer)
VLM_CONFIG = {
    'default_model': 'qwen2.5-vl:7b',
    'max_image_res': 800,           # Standard res for layout logic
    'complex_image_res': 1600,      # High-res for tables/detailed crops
    'timeout_seconds': 30,          # SLA for local/cloud VLM calls
    'distillation_dir': 'research/dataset',
}

# Font Classification Thresholds
FONT_CONFIG = {
    'heading_font_threshold': 14,
    'paragraph_font_threshold': 10,
}

# OCR Configuration
OCR_CONFIG = {
    'use_angle_cls': True,
    'lang': 'en',
    'confidence_threshold': 0.5,
}

# Table Processing Configuration
TABLE_CONFIG = {
    'max_text_boxes': 1000,
    'max_rows': 100,
    'max_cols': 20,
    'y_threshold': 8,  # pixels for row clustering
    'cell_padding': 2,  # padding to avoid line artifacts
    'bbox_padding': 50, # padding around table bbox to capture edge text (increased for borderless tables)
}

# Processing Thresholds (configurable for different PDF types)
PROCESSING_CONFIG = {
    # Region filtering thresholds - adjust based on PDF complexity
    'figure_overlap_threshold': 0.75,  # Keep more text near figures (labels, annotations)
    'table_overlap_threshold': 0.4,    # Filter text inside tables more aggressively
    'base_overlap_threshold': 0.5,     # Default for other region types
    
    # Table validation - filter garbage tables
    'min_table_rows': 1,               # Minimum rows for valid table
    'min_table_cols': 1,               # Minimum columns for valid table
    'min_meaningful_cells': 2,         # Minimum non-empty cells
    'max_empty_cell_ratio': 0.8,       # Max ratio of empty cells (filter if >80% empty)
    'filter_placeholder_cells': True,  # Filter tables with Cell_X_Y or Col1,Col2 only
}

# Output Configuration
OUTPUT_CONFIG = {
    'base_output_dir': 'Output',  # Base directory for all outputs
    'thumbnails_subdir': 'layout_thumbnails',
    'tables_subdir': 'extracted_tables',
    'figures_subdir': 'extracted_figures',  # Separate folder for figures
    'debug_subdir': 'debug_visualizations',
    'json_filename': 'enhanced_layout_blocks.json',
    'markdown_filename': 'extracted_content.md',
}

# Environment Configuration
ENV_CONFIG = {
    'opencv_io_enable_openexr': '0',
    'display': '',  # Disable display for headless operation
}

# DocLayout-YOLO Model Configuration
DOCLAYOUT_CONFIG = {
    'repo_id': 'juliozhao/DocLayout-YOLO-DocStructBench',
    'filename': 'doclayout_yolo_docstructbench_imgsz1024.pt',
    'target_size': 1024,
    'id2label': {
        0: "Title", 1: "Text", 2: "Abandon",
        3: "Figure", 4: "FigureCaption", 5: "Table",
        6: "TableCaption", 7: "TableFootnote",
        8: "IsolatedFormula", 9: "FormulaCaption"
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

def setup_environment():
    """Setup environment variables for headless operation."""
    for key, value in ENV_CONFIG.items():
        os.environ[key.upper()] = value

def get_output_paths(output_dir: str = None):
    """Get standardized output directory paths.
    
    Args:
        output_dir: Output directory path. If None, uses default from config
    
    Returns:
        Dictionary with all output paths
    """
    base_dir = Path(output_dir or OUTPUT_CONFIG['base_output_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': base_dir,
        'thumbnails': base_dir / OUTPUT_CONFIG['thumbnails_subdir'],
        'tables': base_dir / OUTPUT_CONFIG['tables_subdir'],
        'figures': base_dir / OUTPUT_CONFIG['figures_subdir'],
        'debug': base_dir / OUTPUT_CONFIG['debug_subdir'],
        'json_file': base_dir / OUTPUT_CONFIG['json_filename'],
        'markdown_file': base_dir / OUTPUT_CONFIG['markdown_filename'],
    }