---
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
library_name: peft
license: apache-2.0
tags:
- unsloth
- qwen
- qwen2.5
- text-generation
- information-extraction
- trl
- sft
---

# Model Card for librarian-qwen-extractor

This is the designated "Extraction Specialist" for the **Dual-Student Paradigm** in our Master's Thesis on PDF Distillation and Document Intelligence. It acts as the secondary routing engine, transforming raw markdown text grids into strict Pydantic JSON schemas.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "RMunshi/librarian-qwen-extractor"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "Extract the entities from this text into JSON:\nThe ACME 5000 is an industrial router running firmware 2.1."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Procedure & Pipeline Context

This model is fine-tuned explicitly to generate **Chain-of-Thought (CoT)** reasoning logs (`<thought>`) followed by structured JSON (````json````) extraction graphs. 

In our intelligent routing architecture:
1. **Llama-Vision 11B** (Student 1) analyzes document visuals and returns raw formatted grid markdown.
2. **Qwen-2.5 7B** (Student 2 - This Model) digests that markdown and outputs high-fidelity JSON mappings.

### Training Data
Trained on 870 pristine synthetically generated industrial PDF layouts distilled directly using GPT-4o and verified via OpenCV masking. 

### Training Curves
*   **Final Loss**: Convergence at ~0.08
*   **Base Engine**: Qwen 2.5 7B Instruct
*   **Optimization Framework**: Unsloth + Weights & Biases telemetry
