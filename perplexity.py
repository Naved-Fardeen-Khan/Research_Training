import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm

# --- 1. SETTINGS ---
judge_model_id = "openai-community/gpt2"
# Define paths for the three conditions. These were generated in the previous steps and should be updated if the paths change.
paths = {
    "Zeroshot": "combined_csv_files/combined_zero_shot_results.csv",
    "Finetuned": "combined_csv_files/combined_fine_tuned_results.csv",
    "Real World": "combined_csv_files/combined_real_world_results.csv"
}

output_csv = "perplexity_results_comparison.csv"
output_png = "perplexity_comparison.png"
output_png_log = "perplexity_comparison_log.png"
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- 2. LOAD MODEL ---
base_model = AutoModelForCausalLM.from_pretrained(
    judge_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(judge_model_id)

# Function to get perplexity
def get_perplexity(model, sentence):
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return np.nan
    
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    
    ppl = np.exp(loss)

    return ppl

# --- 3. PROCESSING ---
results = []
for label, path in paths.items():
    if not os.path.exists(path):
        print("-" * 50)
        print(f"Warning: {path} not found. Skipping {label}.")
        print("-" * 50)
        continue
    
    print("-" * 50)
    print(f"Processing {label} data from {path}...")
    print("-" * 50)
        
    df = pd.read_csv(path)
    for col in df.columns:
        if "Months" in col:
            age = col.split()[0]  # Extract age from column name
            sentences = df[col].dropna().tolist()
            for sentence in tqdm(sentences, desc=f"Processing {label} - {age} Months"):
                ppl = get_perplexity(base_model, sentence)
                results.append({"Age": age, "Perplexity": ppl, "Condition": label})

# Save Results
df_ppl = pd.DataFrame(results)
df_ppl.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")