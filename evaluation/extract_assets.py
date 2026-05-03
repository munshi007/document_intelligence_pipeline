import os
import json
import shutil

OUTPUT_DIR = "evaluation/assets"
V3_DIR = "output/v3"

def extract_markdown_snippet():
    md_file = os.path.join(V3_DIR, "extracted_content.md")
    if not os.path.exists(md_file):
        print(f"Skipping {md_file}, not found.")
        return
    
    with open(md_file, "r") as f:
        content = f.read()
    
    # Just grab the first 30 lines to show the table structure
    lines = content.split('\n')[:30]
    snippet = '\n'.join(lines) + '\n...'
    
    with open(os.path.join(OUTPUT_DIR, "markdown_snippet.tex"), "w") as f:
        f.write("\\begin{lstlisting}[language=bash, caption={Phase 1 Output: Raw Markdown retaining grid structure.}]\n")
        f.write(snippet)
        f.write("\n\\end{lstlisting}\n")

def extract_json_snippet():
    json_file = os.path.join(V3_DIR, "7000-12121-2251000_universal_extraction.json")
    if not os.path.exists(json_file):
        print(f"Skipping {json_file}, not found.")
        return
        
    with open(json_file, "r") as f:
        data = json.load(f)
        
    # We want to show it's successfully extracting deep parameters
    # Let's slice the parameters array to just 5 items so it fits in a page
    if "parameters" in data and len(data["parameters"]) > 5:
        data["parameters"] = data["parameters"][:5]
        data["parameters"].append({"name": "...", "value": "...", "unit": "..."})
        
    snippet = json.dumps(data, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "json_snippet.tex"), "w") as f:
        f.write("\\begin{lstlisting}[language=json, caption={Phase 2 Output: Structured JSON perfectly mapped from the raw Markdown without token truncation.}]\n")
        f.write(snippet)
        f.write("\n\\end{lstlisting}\n")

def extract_hkg_breadcrumb():
    hkg_file = os.path.join(V3_DIR, "7000-12121-2251000_graph_summary.txt")
    if not os.path.exists(hkg_file):
        print(f"Skipping {hkg_file}, not found.")
        return
        
    with open(hkg_file, "r") as f:
        content = f.read()
        
    # Grab the first node that shows the context
    lines = content.split('\n')
    snippet_lines = []
    found_node = False
    for line in lines:
        if line.startswith("[node_001]"):
            found_node = True
        if found_node:
            snippet_lines.append(line)
            if line.strip() == "": # End of node
                break
                
    snippet = '\n'.join(snippet_lines)
    
    with open(os.path.join(OUTPUT_DIR, "hkg_snippet.tex"), "w") as f:
        f.write("\\begin{lstlisting}[language=bash, caption={Hierarchical Knowledge Graph (HKG) embedding structural breadcrumbs into each node for context-aware retrieval.}]\n")
        f.write(snippet)
        f.write("\n\\end{lstlisting}\n")

def copy_images():
    # Fix: the images are in layout_thumbnails subfolder
    bboxes_img = os.path.join(V3_DIR, "layout_thumbnails/page_02_bboxes.png")
    if os.path.exists(bboxes_img):
        shutil.copy(bboxes_img, os.path.join(OUTPUT_DIR, "yolo_bboxes.png"))
        print(f"Copied {bboxes_img} to {OUTPUT_DIR}/yolo_bboxes.png")
    else:
        # Try page 01 if 02 doesn't exist
        bboxes_img = os.path.join(V3_DIR, "layout_thumbnails/page_01_bboxes.png")
        if os.path.exists(bboxes_img):
            shutil.copy(bboxes_img, os.path.join(OUTPUT_DIR, "yolo_bboxes.png"))
            print(f"Copied {bboxes_img} to {OUTPUT_DIR}/yolo_bboxes.png")
    
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extract_markdown_snippet()
    extract_json_snippet()
    extract_hkg_breadcrumb()
    copy_images()
    print("Assets extracted to evaluation/assets/")
