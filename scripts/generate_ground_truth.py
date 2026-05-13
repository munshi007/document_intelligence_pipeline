#!/usr/bin/env python3
"""
Comprehensive ground truth annotation generator for evaluation dataset.
Extracts structured fields from PDFs for thesis-worthy evaluation benchmarks.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Try importing pdfplumber or fall back to PyPDF2
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    try:
        import PyPDF2
    except ImportError:
        pass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class GroundTruthExtractor:
    """Extract ground truth fields from technical PDFs."""
    
    def __init__(self):
        self.currency_patterns = [
            r'\b(USD|EUR|GBP|CHF|JPY|CNY|INR|MXN)\b',
            r'\$|€|£|¥|¢'
        ]
        
        self.supplier_keywords = [
            'FROM:', 'Supplier:', 'Vendor:', 'Manufacturer:', 'Company:', 'Seller:',
            'from:', 'supplier:', 'vendor:', 'manufacturer:', 'company:', 'seller:',
            'MURRELEKTRONIK', 'murrelektronik'
        ]
        
        self.recipient_keywords = [
            'TO:', 'Customer:', 'Recipient:', 'Client:', 'Buyer:', 'Ship To:',
            'to:', 'customer:', 'recipient:', 'client:', 'buyer:', 'ship to:'
        ]
        
        self.doc_type_keywords = {
            'quotation': ['quotation', 'quote', 'estimate', 'tender', 'bid'],
            'invoice': ['invoice', 'bill', 'receipt', 'account'],
            'technical_specification': ['specification', 'spec', 'datasheet', 'technical data', 'pdf417'],
            'purchase_order': ['purchase order', 'po\b', 'order confirmation'],
            'technical_documentation': ['manual', 'guide', 'technical', 'documentation']
        }
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:10]:  # First 10 pages for speed
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return '\n'.join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber failed on {pdf_path}: {e}")
            return ""
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            text_parts = []
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:10]:  # First 10 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return '\n'.join(text_parts)
        except Exception as e:
            logger.warning(f"PyPDF2 failed on {pdf_path}: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        if HAS_PDFPLUMBER:
            text = self.extract_text_pdfplumber(pdf_path)
            if not text:
                text = self.extract_text_pypdf2(pdf_path)
        else:
            text = self.extract_text_pypdf2(pdf_path)
        return text
    
    def extract_currency(self, text: str) -> Optional[str]:
        """Extract currency code from text."""
        for pattern in self.currency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                currency = match.group(1) if '(' in pattern else match.group(0)
                if currency in ['USD', 'EUR', 'GBP', 'CHF', 'JPY', 'CNY', 'INR', 'MXN']:
                    return currency
                # Map symbols to codes
                symbol_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '¢': 'USD'}
                return symbol_map.get(currency)
        return None
    
    def extract_supplier(self, text: str, filename: str) -> Optional[str]:
        """Extract supplier/vendor name."""
        # Check filename for common supplier patterns
        if 'murrelektronik' in filename.lower() or '7000' in filename:
            return 'Murrelektronik GmbH'
        
        # Search text for supplier indicators
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for keyword in self.supplier_keywords:
                if keyword.lower() in line.lower():
                    # Extract next non-empty line or same line after keyword
                    if ':' in line:
                        supplier = line.split(':', 1)[1].strip()
                    else:
                        supplier = line.replace(keyword, '', re.IGNORECASE).strip()
                    
                    if supplier and len(supplier) > 2:
                        return supplier
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) > 2:
                            return next_line
        
        return None
    
    def extract_recipient(self, text: str) -> Optional[str]:
        """Extract recipient/customer name."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for keyword in self.recipient_keywords:
                if keyword.lower() in line.lower():
                    if ':' in line:
                        recipient = line.split(':', 1)[1].strip()
                    else:
                        recipient = line.replace(keyword, '', re.IGNORECASE).strip()
                    
                    if recipient and len(recipient) > 2:
                        return recipient
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) > 2:
                            return next_line
        
        return None
    
    def extract_document_title(self, text: str, filename: str) -> str:
        """Extract or infer document type/title."""
        text_lower = text.lower()
        
        # Check filename for hints
        if 'quotation' in filename.lower() or 'quote' in filename.lower():
            return 'Technical Quotation'
        if 'invoice' in filename.lower():
            return 'Invoice'
        if 'datasheet' in filename.lower():
            return 'Technical Datasheet'
        
        # Search text for document type keywords
        for doc_type, keywords in self.doc_type_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Map to clean titles
                    title_map = {
                        'quotation': 'Technical Quotation',
                        'invoice': 'Invoice',
                        'technical_specification': 'Technical Specification',
                        'purchase_order': 'Purchase Order',
                        'technical_documentation': 'Technical Documentation'
                    }
                    return title_map.get(doc_type, 'Technical Document')
        
        # Default based on filename patterns
        if 'hdb' in filename.lower():
            return 'HDB Technical Specification'
        if '7000' in filename:
            return 'Murrelektronik Product Specification'
        
        return 'Technical Document'
    
    def extract_parameters(self, text: str) -> List[Dict[str, str]]:
        """Extract technical parameters/specifications."""
        parameters = []
        
        # Look for common specification patterns
        spec_patterns = [
            (r'Supply\s*(?:Voltage|Voltage)?\s*[:\-]?\s*(\d+[^\n]*?V)', 'voltage'),
            (r'(?:Power|Current)\s*[:\-]?\s*(\d+[^\n]*?A)', 'current'),
            (r'(?:Frequency|Hz)\s*[:\-]?\s*(\d+[^\n]*?Hz)', 'frequency'),
            (r'(?:Temperature|Temp)\s*(?:Range)?\s*[:\-]?\s*(\-?\d+[^\n]*?°?C)', 'temperature'),
            (r'(?:Connector|Connection|Type)\s*[:\-]?\s*([^\n]+)', 'connector_type'),
            (r'(?:Material|Housing)\s*[:\-]?\s*([^\n]+)', 'material'),
        ]
        
        for pattern, param_type in spec_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value and len(value) < 100:  # Reasonable parameter length
                    parameters.append({
                        'param_type': param_type,
                        'value': value,
                        'unit': self._extract_unit(value, param_type)
                    })
        
        return parameters[:10]  # Limit to top 10 parameters
    
    def _extract_unit(self, value: str, param_type: str) -> Optional[str]:
        """Extract unit from parameter value."""
        units = {
            'voltage': ['V', 'VAC', 'VDC'],
            'current': ['A', 'mA'],
            'frequency': ['Hz', 'kHz', 'MHz'],
            'temperature': ['°C', 'C', '°F', 'F']
        }
        
        for unit in units.get(param_type, []):
            if unit in value:
                return unit
        
        return None
    
    def extract_standards(self, text: str) -> List[str]:
        """Extract standards and certifications."""
        standards = []
        
        standard_patterns = [
            r'\b(?:ISO|IEC|EN|DIN|ANSI|BS|CSA|UL)\s*[\-:]?\s*\d+[\-:]?\d*[\-:]?\d*',
            r'\bCE\s+(?:Mark|marked|certified)',
            r'\bRoHS',
            r'\bREACH',
            r'\bIP\d{2}',  # IP rating
            r'\bIP\d{2}\-[0-9]',
        ]
        
        for pattern in standard_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            standards.extend(matches)
        
        return list(set(standards))[:8]  # Unique, limit to 8
    
    def extract_connectors(self, text: str) -> List[str]:
        """Extract connector types and component references."""
        connectors = []
        
        connector_keywords = [
            'M12', 'M8', 'M5', 'RJ45', 'RJ11', 'USB', 'HDMI', 'DIN',
            'XLR', 'BNC', 'DBE', 'D-Sub', 'Ethernet', 'Serial',
            'Modular', 'Circular', 'Rectangular', 'Inline'
        ]
        
        for keyword in connector_keywords:
            if keyword.lower() in text.lower():
                connectors.append(keyword)
        
        return list(set(connectors))[:6]  # Unique, limit to 6
    
    def generate_annotation(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        """Generate complete ground truth annotation for a PDF."""
        pdf_name = Path(pdf_path).name
        
        logger.info(f"Extracting: {pdf_name}")
        
        text = self.extract_text(pdf_path)
        
        if not text or len(text) < 50:
            logger.warning(f"Insufficient text extracted from {pdf_name}")
            return None
        
        fields = {}
        
        # Extract core fields
        supplier = self.extract_supplier(text, pdf_name)
        if supplier:
            fields['supplier'] = supplier
        
        recipient = self.extract_recipient(text)
        if recipient:
            fields['recipient'] = recipient
        
        currency = self.extract_currency(text)
        if currency:
            fields['currency'] = currency
        
        fields['document_title'] = self.extract_document_title(text, pdf_name)
        
        # Extract technical fields
        parameters = self.extract_parameters(text)
        if parameters:
            fields['parameters'] = parameters
        
        standards = self.extract_standards(text)
        if standards:
            fields['standards'] = standards
        
        connectors = self.extract_connectors(text)
        if connectors:
            fields['connectors'] = connectors
        
        # Build JSONL entry
        annotation = {
            'doc_id': doc_id,
            'source_pdf': f'data/EVAL_DATA/{pdf_name}',  # Correct path
            'annotated_by': 'automated_extraction',
            'extraction_method': 'pdfplumber_with_regex_parsing',
            'fields': fields,
            'text_length': len(text),
            'field_count': len(fields)
        }
        
        return annotation


def main():
    """Main execution."""
    project_dir = Path(__file__).resolve().parent.parent
    
    cohort_file = project_dir / 'data/eval_lists/eval50_seed42.txt'
    output_file = project_dir / 'data/ground_truth/annotations.jsonl'
    
    if not cohort_file.exists():
        print(f"Cohort file not found: {cohort_file}")
        sys.exit(1)
    
    # Read cohort list (first 25 for initial annotation)
    with open(cohort_file) as f:
        pdf_paths = [line.strip() for line in f.readlines()]
    
    pdf_paths = pdf_paths[:25]  # First 25 for comprehensive ground truth
    
    logger.info(f"Extracting ground truth from {len(pdf_paths)} PDFs...")
    
    extractor = GroundTruthExtractor()
    annotations = []
    
    for idx, pdf_path in enumerate(pdf_paths, 1):
        if not Path(pdf_path).exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        
        annotation = extractor.generate_annotation(pdf_path, str(idx).zfill(5))
        
        if annotation:
            annotations.append(annotation)
            logger.info(f"  ✓ [{idx}/{len(pdf_paths)}] {Path(pdf_path).name} — {annotation['field_count']} fields")
        
        if idx % 5 == 0:
            logger.info(f"  Progress: {idx}/{len(pdf_paths)} completed")
    
    # Write annotations JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')
    
    logger.info(f"\n✅ Generated {len(annotations)} ground truth annotations")
    logger.info(f"📁 Output: {output_file}")
    logger.info(f"\nAnnotation statistics:")
    logger.info(f"  - Total documents: {len(annotations)}")
    logger.info(f"  - Avg fields per doc: {sum(a['field_count'] for a in annotations) / len(annotations):.1f}")
    
    # Summary by field type
    field_types = {}
    for ann in annotations:
        for field_name in ann['fields'].keys():
            field_types[field_name] = field_types.get(field_name, 0) + 1
    
    logger.info(f"\nField coverage:")
    for field_name, count in sorted(field_types.items(), key=lambda x: -x[1]):
        pct = (count / len(annotations)) * 100
        logger.info(f"  - {field_name}: {count}/{len(annotations)} ({pct:.0f}%)")


if __name__ == '__main__':
    main()
