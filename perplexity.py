import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
import os
from tqdm import tqdm

# --- 1. SETTINGS ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-lora/final_adapter"
input_csv = "combined_csv_files/combined_real_world_results.csv"
output_csv = "perplexity_results_comparison.csv"
output_png = "perplexity_comparison.png"
output_png_log = "perplexity_comparison_log.png"
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. LOAD MODEL ---
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Function to get perplexity for a single sentence
def get_perplexity(model, age_label, sentence):
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return np.nan
    
    # 1. Using similar prompt used during training
    # For Phi-3 Instruct, this usually looks like:
    prompt = f"<|user|>\nPredict a sentence a parent says to a child at age {age_label}:<|end|>\n<|assistant|>\n"
    full_text = prompt + sentence + "<|end|>"
    
    # 2. Tokenize the full text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    
    # 3. CRITICAL: Mask the 'Prompt' tokens so we only calculate loss on the 'Response'
    # We don't want to penalize the model for the perplexity of our own instructions!
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    labels[:, :prompt_len] = -100 # -100 is the ignore index for PyTorch loss
    
    # 4. Perform the forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    
    ppl = np.exp(loss)

    return ppl

# --- 3. PROCESSING ---
print("Reading real-world data from CSV...")
df_real = pd.read_csv(input_csv)
results = []

# First: Calculate for Zeroshot (Base Model)
print("-" * 50)
print("Calculating Zeroshot Perplexity...")
print("-" * 50)
for col in tqdm(df_real.columns, desc="Processing Age Groups"):
    sentences = df_real[col].dropna().tolist()
    for sent in sentences:
        ppl = get_perplexity(base_model, col, sent)
        results.append({"Age": col, "Perplexity": ppl, "Condition": "Zeroshot"})

# Second: Load Adapter and Calculate for Finetuned
print("-" * 50)
print("Loading Adapter and Calculating Finetuned Perplexity...")
print("-" * 50)
ft_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

for col in tqdm(df_real.columns, desc="Processing Age Groups"):
    sentences = df_real[col].dropna().tolist()
    for sent in sentences:
        ppl = get_perplexity(ft_model, col, sent)
        results.append({"Age": col, "Perplexity": ppl, "Condition": "Finetuned"})

# Save Results
df_ppl = pd.DataFrame(results)
df_ppl.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")