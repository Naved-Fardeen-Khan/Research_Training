import torch
import matplotlib.pyplot as plt
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig 

# --- 1. CONFIGURATION ---
model_id = "microsoft/Phi-3-mini-4k-instruct"
output_dir = "./phi-3-mini-lora"
print(f"Using model: {model_id}")

# --- 2. LOAD DATA & TOKENIZER ---
print("Loading and splitting dataset...")
raw_dataset = load_dataset("json", data_files="combined.jsonl", split="train")


# Split 10% for validation
train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
eval_dataset = eval_dataset.shuffle(seed=42).select(range(1000))  # Limit eval set to 1,000 samples for quick evaluation
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
tokenizer.padding_side = "right"  # Pad on the right for causal models

# --- 3. FORMATTING THE CHAT TEMPLATE ---
def format_phi3_chat(row):
    chat = [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["output"]}
    ]
    row["text"] = tokenizer.apply_chat_template(chat, tokenize=False)
    return row

formatted_train_dataset = train_dataset.map(format_phi3_chat)
formatted_eval_dataset = eval_dataset.map(format_phi3_chat)

# --- 4. LOAD MODEL & TOKENIZER ---
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager" # For puhti
)

# --- 5. PREPARE MODEL FOR LORA ---
print("Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"] 
)
model = get_peft_model(model, peft_config)

# --- 6. TRAINING ARGUMENTS ---
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch size = 4 * 4 = 16
    learning_rate=1e-5,
    fp16=False,
    optim="paged_adamw_32bit", # Use 32-bit optimizer for stability with 4-bit models
    adam_epsilon=1e-8,
    warmup_ratio=0.03,
    dataset_text_field="text",
    max_length=128,
    report_to="none", # Disable WandB logging for a clean test run
    max_grad_norm=0.3, # Gradient clipping to prevent exploding gradients


    # For 10,000 samples prototype:
    # save_steps=100,
    # save_total_limit=3,
    # logging_steps=50,
    # eval_strategy="steps",
    # eval_steps=100,

    # For complete 800,000 samples:
    save_steps=5000,
    save_total_limit=3,
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=1000,
)

# --- 7. TRAINING ---
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer
)
trainer.train()  # Start fresh, last checkpoint was corrupted

# --- 8. SAVE THE MODEL ---
print("Saving the fine-tuned model...")
trainer.model.save_pretrained(f"{output_dir}/final_adapter")
tokenizer.save_pretrained(f"{output_dir}/final_adapter")
print(f"Success! Model saved to {output_dir}/final_adapter")

# --- 9. PLOT THE PERPLEXITY CURVE ---
print("Extracting logs and generating perplexity curve plot...")

# The trainer stores cross-entropy loss in its log history.

# 1. Open the trainer's hidden diary
log_history = trainer.state.log_history

# 2. Set up our blank lists
train_steps = []
train_loss = []
eval_steps = []
eval_loss = []

# 3. Sort the diary entries into training scores and exam scores
for entry in log_history:
    # Grab training loss (happens every 50 steps based on our config)
    if "loss" in entry and "step" in entry:
        perplexity = math.exp(entry["loss"])
        train_steps.append(entry["step"])
        train_loss.append(perplexity)
    # Grab evaluation loss (happens every 1000 steps based on our config)
    elif "eval_loss" in entry and "step" in entry:
        perplexity = math.exp(entry["eval_loss"])
        eval_steps.append(entry["step"])
        eval_loss.append(perplexity)

# 4. Draw the graph
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label="Training Perplexity", color="blue", alpha=0.6, linewidth=2)

if eval_steps: 
    plt.plot(eval_steps, eval_loss, label="Evaluation (Test) Perplexity", color="red", marker="o", linewidth=2)

plt.title("LoRA Fine-Tuning Perplexity Curve")
plt.xlabel("Training Steps")
plt.ylabel("Perplexity")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# 5. Save the image to your output folder
plot_path = f"{output_dir}/perplexity_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nPerplexity curve saved successfully to {plot_path}")