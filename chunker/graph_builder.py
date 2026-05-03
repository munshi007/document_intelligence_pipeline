"""
GraphBuilder: Librarian-Grade Document Graph Construction
=========================================================
Transforms flat LayoutRegions into a Hierarchical Knowledge Graph (HKG).

Design Pillars (from thesis research):
  1. ATOMIC CHUNKING  – Tables, Figures, Lists are indivisible nodes.
  2. CONTEXT ENRICHMENT – Every node carries its section breadcrumb.
  3. SPATIAL LINKING – Captions ↔ Figures/Tables linked by proximity.
  4. REFERENCE RESOLUTION – "Figure 68" in text → linked to the figure node.
  5. SEMANTIC OVERLAP – Optional trailing context for continuity.

Design References:
  - Docling: Keep semantic boundaries intact, never split a table.
  - mmLayout: Multi-modal graph with typed edges (PARENT_OF, LINKED_TO).
  - Marker: Reading-order preserved from upstream converter.
"""

import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from core.schemas import LayoutRegion, HierarchicalNode, DocumentGraph, RegionType

logger = logging.getLogger(__name__)


# ── Edge types for the document graph ────────────────────────────────
class EdgeType:
    PARENT_OF = "PARENT_OF"
    LINKED_TO = "LINKED_TO"           # Cross-reference (text mentions Fig X)
    CAPTION_OF = "CAPTION_OF"         # Caption ↔ Figure/Table
    CONTINUATION_OF = "CONTINUATION_OF"  # Semantic overlap


class GraphBuilder:
    """
    State machine that builds a Hierarchical Knowledge Graph (HKG).
    """

    # ── Configuration ────────────────────────────────────────────────
    MAX_TEXT_CHUNK = 1200          # Max chars for a text-only node
    OVERLAP_CHARS = 150           # Trailing overlap between text nodes
    CAPTION_PROXIMITY_PX = 80     # Max pixel distance for caption linking

    def __init__(self):
        self.current_stack: List[str] = []   # Active heading hierarchy

    # ── Public API ───────────────────────────────────────────────────
    def build_graph(
        self,
        regions: List[LayoutRegion],
        doc_info: Dict[str, Any],
    ) -> DocumentGraph:
        """
        Three-pass graph construction:
          Pass 1 – Atomic Chunking & Hierarchy Tracking
          Pass 2 – Spatial Linking (Caption ↔ Figure/Table)
          Pass 3 – Reference Resolution ("Fig 68" → node link)
        """
        self.current_stack = []

        # ── Pass 1: Chunking ────────────────────────────────────────
        nodes = self._pass1_chunk(regions)

        # ── Pass 2: Spatial Linking (Caption ↔ nearest Figure/Table)
        self._pass2_spatial_link(nodes)

        # ── Pass 3: Cross-Reference Resolution ──────────────────────
        self._pass3_reference_link(nodes)

        return DocumentGraph(
            doc_id=doc_info.get('doc_id', 'unknown'),
            filename=doc_info.get('filename', 'unknown'),
            nodes=nodes,
            total_pages=doc_info.get('total_pages', 0),
        )

    @staticmethod
    def to_extraction_batches(
        nodes: List[HierarchicalNode],
        max_chars: int = 4000,
        preserve_atomic: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert graph nodes into extraction-ready semantic batches.

        Rules:
        - Keep table/figure nodes atomic when preserve_atomic=True.
        - Group text-like nodes up to max_chars.
        """
        batches: List[Dict[str, Any]] = []
        current_nodes: List[HierarchicalNode] = []
        current_chars = 0

        def flush() -> None:
            nonlocal current_nodes, current_chars
            if not current_nodes:
                return
            batches.append({
                "batch_id": f"batch_{len(batches)+1:04d}",
                "node_ids": [n.node_id for n in current_nodes],
                "contains_table": any(n.metadata.get("contains_table", False) for n in current_nodes),
                "contains_figure": any(n.metadata.get("contains_figure", False) for n in current_nodes),
                "char_count": current_chars,
                "text": "\n\n".join(n.content for n in current_nodes),
            })
            current_nodes = []
            current_chars = 0

        for node in nodes:
            node_text = node.content or ""
            node_chars = len(node_text)
            node_type = (node.metadata.get("node_type") or node.type or "").lower()
            
            # If a single node is already over max_chars, decide whether to split or keep atomic
            if node_chars > max_chars:
                if preserve_atomic:
                    # Original behavior: Keep it atomic even if it's huge
                    flush()
                    batches.append({
                        "batch_id": f"batch_{len(batches)+1:04d}",
                        "node_ids": [node.node_id],
                        "contains_table": bool(node.metadata.get("contains_table", False) or node_type == "table"),
                        "contains_figure": bool(node.metadata.get("contains_figure", False) or node_type == "figure"),
                        "char_count": node_chars,
                        "text": node_text,
                    })
                else:
                    # SHATTER MODE: Force split the massive node into shards
                    flush()
                    # We split based on line boundaries if possible to maintain table structure integrity
                    lines = node_text.split("\n")
                    current_shard_lines = []
                    current_shard_chars = 0
                    
                    for line in lines:
                        if (current_shard_chars + len(line) > max_chars) and current_shard_lines:
                            batches.append({
                                "batch_id": f"batch_{len(batches)+1:04d}",
                                "node_ids": [node.node_id],
                                "contains_table": bool(node.metadata.get("contains_table", False) or node_type == "table"),
                                "char_count": current_shard_chars,
                                "text": "\n".join(current_shard_lines) + "\n(Fragment continued in next batch...)"
                            })
                            current_shard_lines = []
                            current_shard_chars = 0
                        
                        current_shard_lines.append(line)
                        current_shard_chars += len(line) + 1
                    
                    if current_shard_lines:
                        batches.append({
                            "batch_id": f"batch_{len(batches)+1:04d}",
                            "node_ids": [node.node_id],
                            "contains_table": bool(node.metadata.get("contains_table", False) or node_type == "table"),
                            "char_count": current_shard_chars,
                            "text": "\n".join(current_shard_lines)
                        })
                continue

            # Consolidation logic: flush only if this node doesn't fit in the current batch
            if current_nodes and (current_chars + node_chars > max_chars):
                flush()

            current_nodes.append(node)
            current_chars += node_chars

        flush()
        return batches

    # ── Pass 1: Atomic Chunking ──────────────────────────────────────
    def _pass1_chunk(self, regions: List[LayoutRegion]) -> List[HierarchicalNode]:
        nodes: List[HierarchicalNode] = []
        buffer: List[LayoutRegion] = []
        last_text_tail = ""

        def flush():
            nonlocal buffer, last_text_tail
            if not buffer:
                return
            node = self._create_node(
                node_id=f"node_{len(nodes):03d}",
                regions=buffer,
                hierarchy=list(self.current_stack),
                overlap_prefix=last_text_tail,
            )
            nodes.append(node)

            # Capture trailing text for next node's overlap
            text_regions = [r for r in buffer if r.type not in (RegionType.TABLE, RegionType.FIGURE)]
            combined = " ".join((r.text or "") for r in text_regions).strip()
            if len(combined) > self.OVERLAP_CHARS:
                last_text_tail = combined[-self.OVERLAP_CHARS:]
            else:
                last_text_tail = ""

            buffer = []

        for r in regions:
            # ── Rule 1: Tables are ATOMIC (never merged with text) ──
            if r.type == RegionType.TABLE:
                flush()
                buffer.append(r)
                flush()
                continue

            # ── Rule 2: Figures are ATOMIC ───────────────────────────
            if r.type == RegionType.FIGURE:
                flush()
                buffer.append(r)
                flush()
                continue

            # ── Rule 3: Titles / Headings update hierarchy & split ───
            if r.type in (RegionType.TITLE, RegionType.HEADING):
                flush()
                self._update_hierarchy(r)
                buffer.append(r)
                continue

            # ── Rule 4: Captions stay as their own tiny node ─────────
            if r.type == RegionType.CAPTION:
                flush()
                buffer.append(r)
                flush()
                continue

            # ── Rule 5: Footers become their own node ────────────────
            if r.type == RegionType.FOOTER:
                flush()
                buffer.append(r)
                flush()
                continue

            # ── Rule 6: Regular text accumulates in buffer ───────────
            buffer.append(r)
            current_len = sum(len(cr.text or "") for cr in buffer)
            if current_len > self.MAX_TEXT_CHUNK:
                flush()

        flush()
        return nodes

    # ── Pass 2: Spatial Linking (Caption ↔ Figure/Table) ─────────────
    def _pass2_spatial_link(self, nodes: List[HierarchicalNode]):
        """
        Link caption nodes to the nearest figure or table node
        on the same page using bounding-box proximity.
        """
        caption_nodes = [n for n in nodes if n.metadata.get('node_type') == 'caption']
        visual_nodes = [n for n in nodes if n.metadata.get('node_type') in ('table', 'figure')]

        for cap in caption_nodes:
            cap_bbox = cap.metadata.get('bbox')
            cap_page = cap.metadata.get('page')
            if not cap_bbox:
                continue

            best_target = None
            best_dist = float('inf')

            for vis in visual_nodes:
                vis_bbox = vis.metadata.get('bbox')
                vis_page = vis.metadata.get('page')
                if not vis_bbox or vis_page != cap_page:
                    continue

                dist = self._bbox_distance(cap_bbox, vis_bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_target = vis

            if best_target and best_dist < self.CAPTION_PROXIMITY_PX:
                # Bidirectional link
                if best_target.node_id not in cap.linked_nodes:
                    cap.linked_nodes.append(best_target.node_id)
                if cap.node_id not in best_target.linked_nodes:
                    best_target.linked_nodes.append(cap.node_id)
                logger.debug(
                    f"Linked caption '{cap.node_id}' → '{best_target.node_id}' (dist={best_dist:.0f}px)"
                )

    # ── Pass 3: Cross-Reference Resolution ───────────────────────────
    def _pass3_reference_link(self, nodes: List[HierarchicalNode]):
        """
        Scan every node's content for mentions of 'Figure X', 'Fig. X',
        'Table X' and link to the node that contains that entity.
        """
        # Build a lookup: figure/table nodes indexed by their detected IDs
        ref_pattern = re.compile(r'(?:Fig\.?|Figure|Table)\s*(\d+[A-Za-z]*)', re.IGNORECASE)

        # First, identify which nodes ARE figures/tables and extract their IDs
        entity_index: Dict[str, str] = {}  # "68" → "node_002"
        for node in nodes:
            if node.metadata.get('node_type') in ('figure', 'table', 'caption'):
                refs = ref_pattern.findall(node.content)
                for ref_id in refs:
                    entity_index[ref_id.lower()] = node.node_id

        # Second, find all references in text nodes and create links
        for node in nodes:
            refs_in_text = ref_pattern.findall(node.content)
            for ref_id in refs_in_text:
                target_node_id = entity_index.get(ref_id.lower())
                if target_node_id and target_node_id != node.node_id:
                    if target_node_id not in node.linked_nodes:
                        node.linked_nodes.append(target_node_id)
                        logger.debug(
                            f"Cross-ref: '{node.node_id}' mentions ref '{ref_id}' → linked to '{target_node_id}'"
                        )

    # ── Internal Helpers ─────────────────────────────────────────────
    def _update_hierarchy(self, r: LayoutRegion):
        """Update the heading stack based on the region type and content."""
        text = (r.text or "").strip()
        if not text:
            return

        if r.type == RegionType.TITLE:
            self.current_stack = [text]
        elif r.type == RegionType.HEADING:
            # Detect heading level heuristically
            is_lvl1 = text.isupper() and len(text) > 3
            # Numbered sections like "4.6.2" suggest deeper nesting
            has_deep_numbering = bool(re.match(r'^\d+\.\d+\.\d+', text))

            if is_lvl1 or not self.current_stack:
                self.current_stack = [text]
            elif has_deep_numbering and len(self.current_stack) >= 1:
                # Deep section → append as sub-heading
                if len(self.current_stack) >= 3:
                    self.current_stack = [self.current_stack[0], self.current_stack[1], text]
                elif len(self.current_stack) >= 2:
                    self.current_stack = [self.current_stack[0], self.current_stack[1], text]
                else:
                    self.current_stack.append(text)
            elif len(self.current_stack) >= 2:
                self.current_stack = [self.current_stack[0], text]
            else:
                self.current_stack.append(text)

    def _create_node(
        self,
        node_id: str,
        regions: List[LayoutRegion],
        hierarchy: List[str],
        overlap_prefix: str = "",
    ) -> HierarchicalNode:
        """Build a single HierarchicalNode from a list of regions."""
        hierarchy_path = " > ".join(hierarchy) if hierarchy else "(Document Root)"

        # Determine node type
        contains_table = any(r.type == RegionType.TABLE for r in regions)
        contains_figure = any(r.type == RegionType.FIGURE for r in regions)
        is_caption = any(r.type == RegionType.CAPTION for r in regions)
        is_footer = any(r.type == RegionType.FOOTER for r in regions)

        if contains_table:
            node_type = "table"
        elif contains_figure:
            node_type = "figure"
        elif is_caption:
            node_type = "caption"
        elif is_footer:
            node_type = "footer"
        else:
            node_type = "text"

        # ── Build Content ────────────────────────────────────────────
        content_parts: List[str] = []
        content_parts.append(f"[CONTEXT: {hierarchy_path}]")
        content_parts.append("")

        if overlap_prefix:
            content_parts.append(f"... {overlap_prefix} (continued)")
            content_parts.append("")

        # Get primary bbox for spatial metadata
        primary_bbox = regions[0].bbox if regions else [0, 0, 0, 0]

        for r in regions:
            if r.type == RegionType.TABLE:
                table_data = r.metadata.get('table_data', {})
                rows = table_data.get('rows', [])
                anchor = r.metadata.get('anchor_text', '')
                if anchor:
                    content_parts.append(f"**{anchor}**")
                if rows:
                    headers = [str(c) if c else "" for c in rows[0]]
                    content_parts.append("| " + " | ".join(headers) + " |")
                    content_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows[1:]:
                        cells = [str(c) if c else "" for c in row]
                        content_parts.append("| " + " | ".join(cells) + " |")
                else:
                    content_parts.append(r.text or "")

            elif r.type == RegionType.FIGURE:
                fig_path = r.metadata.get('snapshot_image', '')
                content_parts.append(f"[FIGURE: {fig_path}]")
                if r.text:
                    content_parts.append(r.text)

            elif r.type == RegionType.CAPTION:
                content_parts.append(f"Caption: {r.text or ''}")

            elif r.type in (RegionType.TITLE, RegionType.HEADING):
                content_parts.append(f"## {r.text or ''}")

            else:
                if r.text:
                    content_parts.append(r.text.strip())

            content_parts.append("")

        content = "\n".join(content_parts).strip()

        return HierarchicalNode(
            node_id=node_id,
            type=node_type,
            hierarchy=hierarchy,
            content=content,
            metadata={
                "node_type": node_type,
                "contains_table": contains_table,
                "contains_figure": contains_figure,
                "page": regions[0].page if regions else 0,
                "breadcrumb": hierarchy_path,
                "bbox": primary_bbox,
                "region_count": len(regions),
                "region_ids": [r.id for r in regions],
            },
            regions=regions,
        )

    @staticmethod
    def _bbox_distance(bbox_a: List[float], bbox_b: List[float]) -> float:
        """Compute minimum distance between two bounding boxes."""
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return float('inf')
        ax1, ay1, ax2, ay2 = bbox_a[:4]
        bx1, by1, bx2, by2 = bbox_b[:4]

        # If they overlap, distance is 0
        dx = max(ax1 - bx2, bx1 - ax2, 0)
        dy = max(ay1 - by2, by1 - ay2, 0)
        return math.sqrt(dx * dx + dy * dy)
