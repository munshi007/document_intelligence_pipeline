import os
import torch
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer
import wandb

# 1. Configuration
# Resolve project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True
wandb_project = "vlm-distillation"
wandb_entity = "rohanmunshi06-otto-von-guericke-university-magdeburg"

# 2. Initialize Model & Tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth", # Use Unsloth's optimized version
)

# 3. Add LoRA Adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # Finetune Vision Towers
    finetune_language_layers   = True, # Finetune LLM
    finetune_attention_modules = True, # Finetune Attention
    finetune_mlp_modules       = True, # Finetune MLP
    r = 16,           # Rank
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    target_modules = "all-linear", # Target all linear layers
)

# 4. Data Preparation
def format_data(examples):
    conversations = examples["conversations"]
    image_paths = examples["image"]
    
    # Load images as PIL objects
    images = [Image.open(os.path.join(PROJECT_ROOT, p)).convert("RGB") for p in image_paths]
    
    # Reformat conversations for Llama-3.2-Vision
    new_conversations = []
    for conv_list in conversations:
        new_conv = []
        for msg in conv_list:
            role = "user" if msg["from"] == "human" else "assistant"
            value = msg["value"]
            
            if role == "user" and "<image>\n" in value:
                # Remove the <image>\n token and add as a separate content type
                text_content = value.replace("<image>\n", "")
                content = [
                    {"type": "image"},
                    {"type": "text", "text": text_content}
                ]
            else:
                content = [{"type": "text", "text": value}]
            
            new_conv.append({"role": role, "content": content})
        new_conversations.append(new_conv)
    
    return {
        "conversations" : new_conversations,
        "images"        : [[img] for img in images],
    }

# Load the local dataset
dataset_path = os.path.join(PROJECT_ROOT, "research/dataset/dataset.jsonl")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(f"Loaded dataset: {len(dataset)} examples")

def filter_images(example):
    # The image path in dataset.jsonl is "research/dataset/images/..."
    abs_path = os.path.join(PROJECT_ROOT, example["image"])
    return os.path.exists(abs_path)

dataset = dataset.filter(filter_images)
print(f"Dataset after filtering: {len(dataset)} examples")

dataset = dataset.map(format_data, batched = True, batch_size = 50, remove_columns = dataset.column_names)
print(f"Dataset after mapping: {len(dataset)} examples")

if len(dataset) == 0:
    print("ERROR: Dataset is empty after processing!")
    import sys
    sys.exit(1)

# 5. Training Configuration
# Hugging Face: The model is loaded from HF Hub using 'model_name'.
# To save/push to HF: Use model.push_to_hub_merged("your_repo", tokenizer, ...) below.
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Optimized collator
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 1000, # ~2.2 epochs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Log to WandB
        save_strategy = "steps",
        save_steps = 100,
        max_seq_length = max_seq_length,
        dataset_text_field = "", # Leave blank for vision
        dataset_kwargs = {
            "skip_prepare_dataset": True,
        },
        push_to_hub = True, # Push to Hugging Face
        hub_model_id = "vlm-student-thesis", # Repo name
    ),
)

# 6. Start Training
print("Starting training...")
trainer.train()

# 7. Save Model
model.save_pretrained_merged("vlm_student_model", tokenizer, save_method = "merged_16bit")
print("Model saved to vlm_student_model")

# Push to Hugging Face Hub
model.push_to_hub_merged("vlm-student-thesis", tokenizer, save_method = "merged_16bit")
print("Model pushed to Hugging Face: vlm-student-thesis")
