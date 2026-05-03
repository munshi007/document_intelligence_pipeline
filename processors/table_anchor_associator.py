import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TableAnchorAssociator:
    """
    Associates text labels (anchors/titles) that lie outside a table's bounding box
    with the table itself, preventing them from being orphaned in the reading order.
    
    Handles multi-line text blocks by extracting relevant lines for each table.
    """
    
    def __init__(self, search_distance: float = 120.0, horizontal_tolerance: float = 50.0):
        """
        Initialize the associator.
        
        Args:
            search_distance: Max pixel distance to search above/left of a table.
            horizontal_tolerance: Max pixel difference for left alignment check.
        """
        self.search_distance = search_distance
        self.horizontal_tolerance = horizontal_tolerance
        
    def _extract_best_lines_for_table(self, text: str, text_bbox: List[float], table_bbox: List[float]) -> Optional[str]:
        """
        Extract the most relevant line(s) from a multi-line text block for a specific table.
        
        Strategy:
        - Split text into non-empty lines
        - Find lines that are closest vertically to the table top
        - Extract the most relevant semantic lines (exclude "Pin-assignment" if other lines are closer)
        
        Args:
            text: Full text block (may contain multiple lines)
            text_bbox: [x1, y1, x2, y2] of text block
            table_bbox: [x1, y1, x2, y2] of table
            
        Returns:
            Best anchor text for this table, or None
        """
        if not text or not text.strip():
            return None
            
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
            
        # If single line or short text, return it
        if len(lines) <= 1:
            return text.strip()
            
        # Multi-line case: try to associate lines based on proximity
        _, t_y1, _, _ = table_bbox
        _, text_y1, _, text_y2 = text_bbox
        
        # Calculate approximate line heights and positions
        text_height = text_y2 - text_y1
        line_height = text_height / len(lines)
        
        # For each line, estimate its Y position
        line_data = []
        for i, line in enumerate(lines):
            # Estimate vertical position of this line within the text block
            estimated_y = text_y1 + (i + 0.5) * line_height
            distance_to_table = abs(estimated_y - t_y1)
            line_data.append((distance_to_table, i, line))
            
        # Sort by distance to table top
        line_data.sort(key=lambda x: x[0])
        
        # Get the closest line(s)
        closest_distance, _, closest_line = line_data[0]
        
        # Strategy: Get the closest line (typically "System bus input port" etc.)
        # and at most 1-2 following lines (e.g., "M12 male connector A-coded").
        # This prevents extracting the entire text block.
        closest_distance, closest_idx, closest_line = line_data[0]
        
        # Collect best lines within a TIGHT vertical window
        anchor_lines = [closest_line]
        
        # Search the lines immediately following the closest line (idx > closest_idx)
        # to preserve natural reading order
        following_lines = [item for item in line_data if item[1] > closest_idx]
        following_lines.sort(key=lambda x: x[1]) # sort by index
        
        for distance, idx, line in following_lines:
            # Only include if: very close vertically to the top line AND not empty AND descriptive
            if abs(distance - closest_distance) <= 100:  # Allow small spacing between lines
                if line and len(line) >= 3 and not line.startswith('Pin'):
                    anchor_lines.append(line)
                    if len(anchor_lines) >= 2:  # Limit to 2 lines max
                        break
            else:
                # If gap is too large, break
                if abs(distance - closest_distance) > 50:
                    break
                
        return '\n'.join(anchor_lines)
    def associate_anchors(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scan for standalone text blocks near tables and attach them to the tables.
        
        For multi-line text blocks, extracts the most relevant line(s) for each table
        without consuming the entire block.
        
        Args:
            regions: Processed regions list (e.g., after reading order and merging).
            
        Returns:
            A new list of regions with anchors attached to tables and removed as standalone text.
        """
        if not regions:
            return regions
            
        tables = [r for r in regions if r.get('type', '').lower() in ['table']]
        text_regions = [r for r in regions if r.get('type', '').lower() in ['text', 'title', 'heading']]
        
        if not tables or not text_regions:
            return regions
            
        logger.info(f"Associating anchors for {len(tables)} tables among {len(text_regions)} text blocks.")
        
        # Track text regions that should be completely removed (non-multi-line matches)
        fully_absorbed_ids = set()
        
        for table in tables:
            table_bbox = table.get('bbox')
            if not table_bbox:
                continue
                
            t_x1, t_y1, t_x2, t_y2 = table_bbox
            
            # Find the best candidate text block
            best_anchor = None
            min_dist = float('inf')
            
            for text_reg in text_regions:
                # Skip if already fully absorbed
                if text_reg.get('region_id') in fully_absorbed_ids or text_reg.get('id') in fully_absorbed_ids:
                    continue
                    
                text_bbox = text_reg.get('bbox')
                if not text_bbox:
                    continue
                    
                c_x1, c_y1, c_x2, c_y2 = text_bbox
                
                # The text bottom (c_y2) should be above or slightly inside the table top (t_y1)
                vertical_gap = t_y1 - c_y2
                
                # Check vertical overlap indicating the text is physically parallel to the table body
                vertical_overlap = min(t_y2, c_y2) - max(t_y1, c_y1)
                
                # Check horizontal position
                horizontal_offset = abs(c_x1 - t_x1)
                
                # Condition 1: Text positioned strictly above the table
                is_above = (0 <= vertical_gap <= self.search_distance) or (-20 <= vertical_gap <= 0)
                
                # Condition 2: Text positioned strictly parallel to the table (overlap > 0)
                is_parallel = vertical_overlap > 0
                
                # Combine Y-axis conditions
                is_valid_y = is_above or is_parallel
                
                # Combine X-axis conditions
                is_left_aligned = horizontal_offset <= self.horizontal_tolerance
                is_to_the_left = c_x2 <= t_x1 + 30 # ends before or cleanly near the start of the table
                
                is_valid_x = is_left_aligned or is_to_the_left
                
                if is_valid_y and is_valid_x:
                    # Calculate custom distance metric depending on whether it's above or parallel
                    if is_parallel:
                        # Parallel distance is primarily horizontal
                        dist = max(0, t_x1 - c_x2) 
                        dist += abs(t_y1 - c_y1) * 0.5 # Add small penalty for not being top-aligned
                    else:
                        # Top-aligned distance is primarily vertical
                        dist = (vertical_gap ** 2 + horizontal_offset ** 2) ** 0.5
                        
                    if dist < min_dist:
                        min_dist = dist
                        best_anchor = text_reg
            
            # If no match with strict horizontal criteria, try again with relaxed criteria for parallel blocks
            if best_anchor is None:
                for text_reg in text_regions:
                    if text_reg.get('region_id') in fully_absorbed_ids or text_reg.get('id') in fully_absorbed_ids:
                        continue
                        
                    text_bbox = text_reg.get('bbox')
                    if not text_bbox:
                        continue
                        
                    c_x1, c_y1, c_x2, c_y2 = text_bbox
                    
                    vertical_gap = t_y1 - c_y2
                    vertical_overlap = min(t_y2, c_y2) - max(t_y1, c_y1)
                    
                    # For parallel text blocks (overlapping Y), relax horizontal constraints
                    # This handles multi-column layouts where text is in one column and tables in another
                    if vertical_overlap > 0:  # Has vertical overlap (parallel)
                        # Accept if text is positioned to the left and reasonably nearby
                        if c_x2 <= t_x2 + 200:  # Text ends before or near table right edge
                            # Calculate distance allowing for column separation
                            dist = max(0, t_x1 - c_x2)  # Horizontal gap
                            dist += abs(t_y1 - c_y1) * 0.1  # Small vertical penalty
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_anchor = text_reg

            if best_anchor:
                # We found an anchor for this table
                full_text = best_anchor.get('text', '').strip()
                
                # Check if this is a multi-line text block (contains multiple semantic labels)
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                is_multiline = len(lines) > 1
                
                if is_multiline:
                    # Extract the most relevant line(s) for this specific table
                    anchor_text = self._extract_best_lines_for_table(full_text, text_bbox, table_bbox)
                    
                    if anchor_text:
                        # Remove the consumed text from the region so it doesn't print as a duplicate
                        consumed_lines = [c.strip() for c in anchor_text.split('\n')]
                        remaining_lines = []
                        for line in full_text.split('\n'):
                            if line.strip() not in consumed_lines:
                                remaining_lines.append(line)
                        best_anchor['text'] = '\n'.join(remaining_lines)
                        
                        # If remaining block is virtually empty, mark fully absorbed
                        if not best_anchor['text'].strip():
                            absorbed_id = best_anchor.get('region_id') or best_anchor.get('id')
                            fully_absorbed_ids.add(absorbed_id)
                else:
                    # Single-line anchor - consume the entire block
                    anchor_text = full_text
                    absorbed_id = best_anchor.get('region_id') or best_anchor.get('id')
                    fully_absorbed_ids.add(absorbed_id)
                
                if anchor_text:
                    # Collect existing anchor texts if multiple passes occur
                    existing_anchor = table.get('anchor_text', '')
                    if existing_anchor:
                        table['anchor_text'] = f"{anchor_text}\n{existing_anchor}"
                    else:
                        table['anchor_text'] = anchor_text
                        
                    logger.info(f"Associated anchor '{anchor_text[:60]}...' to table {table.get('id', table.get('region_id'))}")

        logger.info(f"Table Anchor Association complete. Fully_absorbed={len(fully_absorbed_ids)} blocks")
        
        # Reconstruct the regions array excluding fully absorbed text blocks
        # But KEEP multi-line blocks (they may have been partially used)
        final_regions = []
        for r in regions:
            r_id = r.get('region_id') or r.get('id')
            if r_id not in fully_absorbed_ids:
                final_regions.append(r)
            else:
                logger.info(f"Removing fully-absorbed text block: {r_id}")
                
        return final_regions
