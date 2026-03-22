from unsloth import FastVisionModel
import torch
from PIL import Image
import os

# Using the latest available checkpoint
checkpoint_path = "/home/rmunshi/PROJECT/TEST/PROJECTS/pdf_pipeline copy/research/outputs/checkpoint-800"

print(f"Loading model from {checkpoint_path}...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = checkpoint_path,
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)

# Test Image from dataset
image_path = "/home/rmunshi/PROJECT/TEST/PROJECTS/pdf_pipeline copy/research/dataset/images/crop_acdd3c35d600e34e.jpg"
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    import sys
    sys.exit(1)

image = Image.open(image_path).convert("RGB")

instruction = "Analyze the layout and typography of this document page. Identify the visual hierarchy: title, headers (H1, H2, H3), body text, and captions. For each style, estimate its relative font size (e.g., body=12, H1=18), whether it is bold or italic, and note any distinct colors. Do not hallucinate content; focus purely on the visual structure."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

print("\n--- MODEL OUTPUT ---")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)

_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.5, # Lower temperature for stable thesis JSON output
)
print("--- END OUTPUT ---\n")
