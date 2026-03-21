"""
Result Builder Module
Handles construction of final result dictionaries, statistics aggregation, and saving results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from config import OUTPUT_CONFIG

logger = logging.getLogger(__name__)

def count_region_types(regions: List[Dict]) -> Dict[str, int]:
    """Count regions by type."""
    counts = {}
    for region in regions:
        region_type = region.get('type', 'unknown')
        counts[region_type] = counts.get(region_type, 0) + 1
    return counts

def count_processing_methods(regions: List[Dict]) -> Dict[str, int]:
    """Count regions by processing method."""
    counts = {}
    for region in regions:
        source = region.get('source', 'unknown')
        counts[source] = counts.get(source, 0) + 1
    return counts

def aggregate_stats(pages_data: List[Dict], stat_key: str) -> Dict[str, int]:
    """Aggregate statistics across all pages."""
    aggregated = {}
    for page_data in pages_data:
        stats = page_data.get("stats", {}).get(stat_key, {})
        for key, value in stats.items():
            aggregated[key] = aggregated.get(key, 0) + value
    return aggregated

def generate_summary(pages_data: List[Dict]) -> Dict[str, Any]:
    """Generate processing summary."""
    return {
        "total_regions": sum(len(p.get("regions", [])) for p in pages_data),
        "pages_processed": len(pages_data),
        "pages_with_errors": len([p for p in pages_data if "error" in p]),
        "tables_found": sum(p.get("stats", {}).get("tables_found", 0) for p in pages_data),
        "region_types": aggregate_stats(pages_data, "region_types"),
        "processing_methods": aggregate_stats(pages_data, "processing_methods")
    }

def build_final_result(pdf_path: str, doc_page_count: int, pages_data: List[Dict], all_markdown: List[str], config: Dict) -> Dict[str, Any]:
    """Build the final result dictionary."""
    
    summary = generate_summary(pages_data)
    
    result = {
        "document_info": {
            "path": pdf_path,
            "filename": Path(pdf_path).name,
            "total_pages": doc_page_count,
            "processing_timestamp": datetime.now().isoformat()
        },
        "pipeline_config": config,
        "model_info": {
            "layout_model_available": True,
            "ocr_reader_available": True,
            "ocr_type": "paddle",
            "dependencies": {
                "doclayout_yolo": True,
                "torch": True,
                "ultralytics": True,
                "layoutparser": True
            }
        },
        "pages": pages_data,
        "full_markdown": "\n\n".join(all_markdown),
        "summary": summary
    }
    return result

def save_results(result: Dict[str, Any], output_paths: Dict[str, Path]):
    """Save processing results to files."""

    # Custom JSON encoder to handle Pydantic models (like FontSignature)
    def pydantic_encoder(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    # Save JSON results
    try:
        with open(output_paths['json_file'], 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=pydantic_encoder)
        logger.info(f"Results saved to: {output_paths['json_file']}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save markdown content
    try:
        with open(output_paths['markdown_file'], 'w', encoding='utf-8') as f:
            f.write(result['full_markdown'])
        logger.info(f"Markdown content saved to: {output_paths['markdown_file']}")
    except Exception as e:
        logger.error(f"Failed to save Markdown content: {e}")
