"""
Storage: Ephemeral Structural Store (ESS)
A local knowledge store for managing Graph-Native document nodes.
Optimized for high-precision retrieval and schema-based browsing.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.schemas import DocumentGraph, HierarchicalNode

logger = logging.getLogger(__name__)

class EphemeralStore:
    """
    Manages the persistent storage and retrieval of DocumentGraphs.
    Acts as the 'Structural Fact Store' for the Agentic Extractor.
    """
    
    def __init__(self, storage_dir: str = "output/v3/storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_graph: Optional[DocumentGraph] = None
        
    def save_graph(self, graph: DocumentGraph):
        """
        Saves a DocumentGraph to disk as a structured JSON file.
        """
        output_file = self.storage_dir / f"{graph.doc_id}_graph.json"
        
        # We manually serialize to ensure Pydantic to_dict is used
        with open(output_file, 'w', encoding='utf-8') as f:
            # Pydantic v2 .model_dump()
            json.dump(graph.model_dump(), f, indent=4, ensure_ascii=False)
            
        self.active_graph = graph
        logger.info(f"Graph saved to ESS: {output_file}")
        
    def load_graph(self, doc_id: str) -> Optional[DocumentGraph]:
        """
        Loads a DocumentGraph from the ESS store.
        """
        input_file = self.storage_dir / f"{doc_id}_graph.json"
        if not input_file.exists():
            return None
            
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            graph = DocumentGraph(**data)
            self.active_graph = graph
            return graph

    def get_nodes_by_type(self, node_type: str) -> List[HierarchicalNode]:
        """
        Returns all nodes of a specific type (e.g., 'table_centric').
        """
        if not self.active_graph:
            return []
        return [n for n in self.active_graph.nodes if node_type in n.type]

    def get_neighbors(self, node_id: str) -> List[HierarchicalNode]:
        """
        Returns the neighbors of a node (linked cross-references).
        """
        if not self.active_graph:
            return []
            
        source_node = next((n for n in self.active_graph.nodes if n.node_id == node_id), None)
        if not source_node:
            return []
            
        neighbors = []
        for lid in source_node.linked_nodes:
            neighbor = next((n for n in self.active_graph.nodes if n.node_id == lid), None)
            if neighbor:
                neighbors.append(neighbor)
                
        return neighbors

    def get_all_context(self) -> str:
        """
        Returns a single flattened string of all nodes for large-context models.
        """
        if not self.active_graph:
            return ""
        
        full_text = ""
        for n in self.active_graph.nodes:
            full_text += f"-- Node: {n.node_id} (Page {n.metadata.get('page')})\n"
            full_text += f"Hierarchy: {' > '.join(n.hierarchy)}\n"
            full_text += n.content + "\n\n"
            
        return full_text
