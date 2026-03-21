import os
from datasets import load_dataset
from PIL import Image

dataset_path = "/home/rmunshi/PROJECT/TEST/PROJECTS/pdf_pipeline copy/research/dataset/dataset.jsonl"
base_dir = "/home/rmunshi/PROJECT/TEST/PROJECTS/pdf_pipeline copy"

print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(f"Total entries: {len(dataset)}")

# Check first 5 entries
for i in range(min(5, len(dataset))):
    item = dataset[i]
    img_path = os.path.join(base_dir, item["image"])
    exists = os.path.exists(img_path)
    print(f"[{i}] Image: {item['image']} - Exists: {exists}")
    if exists:
        try:
            with Image.open(img_path) as img:
                print(f"    Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            print(f"    Error opening image: {e}")

print("\nVerifying conversation format:")
print(dataset[0]["conversations"])
