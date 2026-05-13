#!/usr/bin/env python3
"""
Generate ground truth annotations using GPT-4o vision for high-quality labels.
This script reads the first 25 PDFs from eval50_seed42.txt and uses GPT-4o to extract structured fields.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import base64
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.error("openai package not installed: pip install openai")
    sys.exit(1)


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into the environment."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class GPTGroundTruthGenerator:
    """Generate ground truth using GPT-4o vision."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
    
    def encode_pdf_page(self, pdf_path: str, page_num: int = 1) -> Optional[str]:
        """Encode first page of PDF to base64."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if page_num > len(pdf.pages):
                    page_num = 1
                page = pdf.pages[page_num - 1]
                # Get image of page
                im = page.to_image()
                # Convert to base64
                import io
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to encode PDF page: {e}")
            return None
    
    def extract_fields_with_gpt(self, pdf_path: str, doc_id: str, image_b64: str) -> Optional[Dict[str, Any]]:
        """Extract fields from PDF page using GPT-4o vision."""
        
        prompt = """Analyze this technical document (PDF page) and extract the following fields in JSON format:
{
  "supplier": "Company/vendor name if visible",
  "recipient": "Customer/recipient name if visible in TO/shipping section",
  "currency": "Currency code (USD, EUR, etc.) if mentioned",
  "document_title": "Document type (e.g., Invoice, Quotation, Technical Spec, Datasheet)",
  "parameters": ["List of key technical specifications (voltage, current, temperature, etc.) if present"],
  "standards": ["Certifications/standards (ISO, CE, RoHS, FCC, UL, etc.) if mentioned"],
  "connectors": ["Connector types (M12, RJ45, M8, DIN, etc.) if visible"],
  "product_type": "Product category if identifiable"
}

Return ONLY valid JSON with no markdown. Extract only fields that are clearly visible in the document.
Do NOT guess or invent values. Use null for missing fields."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON
            try:
                fields = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    fields = json.loads(match.group(0))
                else:
                    logger.warning(f"Could not parse GPT response for {doc_id}")
                    return None
            
            # Clean up fields
            cleaned = {}
            for key, val in fields.items():
                if val is not None and val != "":
                    if isinstance(val, list):
                        cleaned[key] = [v for v in val if v]
                    else:
                        cleaned[key] = val
            
            return cleaned
        
        except Exception as e:
            logger.error(f"GPT-4o API error for {doc_id}: {e}")
            return None
    
    def generate_annotation(self, pdf_path: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Generate complete ground truth annotation using GPT-4o."""
        pdf_name = Path(pdf_path).name
        logger.info(f"[GPT-4o] Processing: {pdf_name}")
        
        # Encode first page
        image_b64 = self.encode_pdf_page(pdf_path)
        if not image_b64:
            logger.warning(f"Could not encode page for {pdf_name}")
            return None
        
        # Extract fields with GPT
        fields = self.extract_fields_with_gpt(pdf_path, doc_id, image_b64)
        if not fields:
            logger.warning(f"GPT extraction failed for {pdf_name}")
            return None
        
        # Build annotation
        annotation = {
            "doc_id": doc_id,
            "source_pdf": f"data/EVAL_DATA/{pdf_name}",
            "annotated_by": "gpt-4o_vision",
            "extraction_method": "gpt_4o_vision_api",
            "fields": fields,
            "field_count": len(fields)
        }
        
        logger.info(f"  ✓ {pdf_name} — {len(fields)} fields extracted")
        return annotation


def main():
    """Main execution."""
    project_dir = Path(__file__).resolve().parents[1]
    load_env_file(project_dir / '.env')
    
    cohort_file = project_dir / 'data/eval_lists/eval50_seed42.txt'
    output_file = project_dir / 'data/ground_truth/annotations_gpt4o.jsonl'
    
    if not cohort_file.exists():
        logger.error(f"Cohort file not found: {cohort_file}")
        sys.exit(1)
    
    # Read cohort list (first 25 for annotation)
    with open(cohort_file) as f:
        pdf_paths = [line.strip() for line in f.readlines()]
    
    pdf_paths = pdf_paths[:25]  # First 25
    
    logger.info(f"Generating GPT-4o ground truth for {len(pdf_paths)} PDFs...")
    logger.info("(This uses the OpenAI API; costs ~$0.50-1.00 total)\n")
    
    generator = GPTGroundTruthGenerator()
    annotations = []
    failed = []
    
    for idx, pdf_path in enumerate(pdf_paths, 1):
        if not Path(pdf_path).exists():
            logger.warning(f"PDF not found: {pdf_path}")
            failed.append(Path(pdf_path).name)
            continue
        
        annotation = generator.generate_annotation(pdf_path, str(idx).zfill(5))
        
        if annotation:
            annotations.append(annotation)
        else:
            failed.append(Path(pdf_path).name)
        
        if idx % 5 == 0:
            logger.info(f"  Progress: {idx}/{len(pdf_paths)} completed")
    
    # Write annotations
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')
    
    logger.info(f"\n✅ Generated {len(annotations)} GPT-4o ground truth annotations")
    logger.info(f"📁 Output: {output_file}")
    
    if failed:
        logger.info(f"⚠️  Failed: {len(failed)} documents")
    
    # Field coverage stats
    field_types = {}
    for ann in annotations:
        for field_name in ann['fields'].keys():
            field_types[field_name] = field_types.get(field_name, 0) + 1
    
    logger.info(f"\nField coverage:")
    for field_name, count in sorted(field_types.items(), key=lambda x: -x[1]):
        pct = (count / len(annotations)) * 100 if annotations else 0
        logger.info(f"  - {field_name}: {count}/{len(annotations)} ({pct:.0f}%)")


if __name__ == '__main__':
    main()
