import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTPUT_IMG = "evaluation/assets/hkg_topology.png"

def visualize_hkg():
    G = nx.DiGraph()
    
    # Neo4j Vibrant Palette
    COLORS = {
        "Product": "#D946EF",   # Fuchsia
        "Attribute": "#FBBF24", # Amber
        "Component": "#3B82F6", # Blue
        "Connector": "#F43F5E", # Rose
        "Pin": "#10B981",       # Emerald
        "Signal": "#8B5CF6"     # Violet
    }
    
    # Root Node
    root = "Cube67+ DIO16"
    G.add_node(root, label="Cube67+", type="Product")

    # Product Attributes
    G.add_node("mfg", label="Murrelektronik", type="Attribute")
    G.add_edge(root, "mfg")
    G.add_node("art", label="Art: 56768", type="Attribute")
    G.add_edge(root, "art")

    # Define the complex hierarchy (simulating the full un-truncated extraction)
    structure = {
        "System Bus Input": {
            "type": "M12 Male",
            "coding": "A-coded",
            "pins": {
                "Pin 1": "24V UA", "Pin 2": "24V US", "Pin 3": "0V",
                "Pin 4": "BUS IN", "Pin 5": "BUS IN", "Pin 6": "0V"
            }
        },
        "System Bus Output": {
            "type": "M12 Female",
            "coding": "A-coded",
            "pins": {
                "Pin 1": "24V UA", "Pin 2": "24V US", "Pin 3": "0V",
                "Pin 4": "BUS OUT", "Pin 5": "BUS OUT", "Pin 6": "0V"
            }
        },
        "Multifunctional Port": {
            "type": "M12 Female (AUX)",
            "coding": "A-coded",
            "pins": {
                "Pin 1": "Sensor 24V", "Pin 2": "DI/DO", "Pin 3": "0V",
                "Pin 4": "DI/DO", "Pin 5": "FE"
            }
        }
    }

    # Build the graph
    for port_name, port_data in structure.items():
        # Port Node (Component)
        G.add_node(port_name, label=port_name, type="Component")
        G.add_edge(root, port_name)
        
        # Connector Node
        conn_id = f"{port_name}_conn"
        G.add_node(conn_id, label=port_data["type"], type="Connector")
        G.add_edge(port_name, conn_id)
        
        # Coding Attribute
        cod_id = f"{port_name}_coding"
        G.add_node(cod_id, label=port_data["coding"], type="Attribute")
        G.add_edge(conn_id, cod_id)
        
        # Pins and Signals
        for pin_name, signal in port_data["pins"].items():
            pin_id = f"{port_name}_{pin_name}"
            G.add_node(pin_id, label=pin_name, type="Pin")
            G.add_edge(conn_id, pin_id)
            
            sig_id = f"{pin_id}_{signal}"
            # Check if signal node already exists to create interconnectivity (like true knowledge graph)
            if not G.has_node(sig_id):
                G.add_node(sig_id, label=signal, type="Signal")
            G.add_edge(pin_id, sig_id)
            
            # Shared Ground Node (Interconnectivity)
            if "0V" in signal:
                if not G.has_node("Common_GND"):
                    G.add_node("Common_GND", label="Common GND", type="Signal")
                G.add_edge(pin_id, "Common_GND")

    # Layout: Force-directed
    plt.figure(figsize=(20, 14), facecolor='#FAFAFA')
    
    # Spring layout with careful parameters to create distinct clusters
    pos = nx.spring_layout(G, k=0.35, iterations=300, seed=100)
    
    node_colors = [COLORS.get(G.nodes[n]['type'], "#9CA3AF") for n in G.nodes()]
    node_sizes = []
    for n in G.nodes():
        ntype = G.nodes[n]['type']
        if ntype == "Product": node_sizes.append(4000)
        elif ntype == "Component": node_sizes.append(2500)
        elif ntype == "Connector": node_sizes.append(1800)
        elif ntype == "Pin": node_sizes.append(800)
        else: node_sizes.append(1200)
        
    labels = nx.get_node_attributes(G, 'label')

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="#CBD5E1", width=1.5, alpha=0.6, 
                           arrowsize=12, connectionstyle="arc3,rad=0.05")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.9, edgecolors="#FFFFFF", linewidths=2.5)
    
    # Draw labels
    for node, (x, y) in pos.items():
        plt.text(x, y, labels[node], fontsize=9, fontweight='bold', 
                 ha='center', va='center', color="#0F172A",
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()]
    plt.legend(handles=patches, loc='lower left', frameon=True, fontsize=14, title="Ontology Entities", title_fontsize=16)

    plt.title("Librarian HKG: Clustered Component Ontology", 
              fontsize=28, fontweight='bold', color="#0F172A", pad=20)
    
    plt.axis('off')
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Clustered Neo4j-style visualization saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    visualize_hkg()
