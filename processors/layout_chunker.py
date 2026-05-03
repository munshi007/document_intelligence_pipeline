"""
Layout-Aware Semantic Chunker
Intelligently groups LayoutRegions into coherent semantic chunks for LLM processing.
Unlike token-based chunkers, this respects Tables, Figures, and Headings as atomic boundaries.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LayoutChunker:
    """
    Chunks a list of regions into sub-lists based on layout and semantic boundaries.
    """
    
    def __init__(self, max_chars: int = 2500):
        self.max_chars = max_chars
        
    def chunk_regions(self, regions: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
        """
        Groups regions into chunks.
        Rules:
        1. Never split a Table or Figure.
        2. Break at Headings if the current chunk is sufficiently large.
        3. Break after text regions if the chunk exceeds max_chars.
        """
        chunks = []
        current_regions = []
        current_chars = 0
        chunk_idx = 1
        section_stack: List[str] = []
        
        def finalize_chunk():
            nonlocal current_regions, current_chars, chunk_idx
            if current_regions:
                region_ids = []
                linked_region_ids = []
                contains_table = False
                for region in current_regions:
                    rid = region.get('region_id') or region.get('id')
                    if rid:
                        region_ids.append(rid)
                    r_type = (region.get('type') or '').lower()
                    if r_type == 'table':
                        contains_table = True
                    linked = region.get('linked_region_ids') or region.get('linked_nodes') or []
                    if isinstance(linked, list):
                        linked_region_ids.extend([str(x) for x in linked])

                chunks.append({
                    "chunk_id": f"chunk_p{page_num}_{chunk_idx:02d}",
                    "page_num": page_num,
                    "regions": current_regions,
                    "char_count": current_chars,
                    "section_path": " > ".join(section_stack) if section_stack else "(Document Root)",
                    "region_ids": region_ids,
                    "contains_table": contains_table,
                    "linked_region_ids": sorted(set(linked_region_ids)),
                })
                chunk_idx += 1
                current_regions = []
                current_chars = 0

        for r in regions:
            r_type = r.get('type', '').lower()
            text_content = r.get('text', '') or ''
            # Adding table/anchor text to char count estimation
            if r_type == 'table':
                text_content += r.get('anchor_text', '') + str(r.get('table_data', ''))
                
            char_len = len(text_content)
            
            # Rule 2: Break at Headings if current chunk is somewhat large
            if r_type in ['title', 'heading']:
                title_text = (r.get('text') or '').strip()
                if title_text:
                    if r_type == 'title':
                        section_stack = [title_text]
                    elif section_stack:
                        section_stack = [section_stack[0], title_text]
                    else:
                        section_stack = [title_text]
                if current_chars > (self.max_chars * 0.5):
                    logger.debug(f"Chunking at {r_type} because current_chars={current_chars} > {self.max_chars * 0.5}")
                    finalize_chunk()
            
            # Add to current chunk
            current_regions.append(r)
            current_chars += char_len
            
            # Rule 3: Break after text if we exceeded limit (and the region wasn't a table/figure)
            if current_chars >= self.max_chars and r_type not in ['table', 'figure']:
                logger.debug(f"Chunking after text because current_chars={current_chars} >= {self.max_chars}")
                finalize_chunk()
                
        # Finalize any remainder
        finalize_chunk()
        
        return chunks
