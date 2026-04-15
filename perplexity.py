import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
import os

# --- 1. SETTINGS ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-lora/final_adapter"
input_csv = "combined_csv_files/combined_real_world_results.csv"
output_csv = "perplexity_results_comparison.csv"
output_png = "perplexity_comparison.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. LOAD MODELS ---
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)

# Function to get perplexity for a single sentence
def get_perplexity(model, text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return np.exp(loss) # PPL = e^loss

# --- 3. PROCESSING ---
df_real = pd.read_csv(input_csv)
results = []

# First: Calculate for Zeroshot (Base Model)
print("Calculating Zeroshot Perplexity...")
for col in df_real.columns:
    sentences = df_real[col].dropna().tolist()
    for sent in sentences:
        ppl = get_perplexity(base_model, sent)
        results.append({"Age": col, "Perplexity": ppl, "Condition": "Zeroshot"})

# Second: Load Adapter and Calculate for Finetuned
print("Loading Adapter and Calculating Finetuned Perplexity...")
ft_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

for col in df_real.columns:
    sentences = df_real[col].dropna().tolist()
    for sent in sentences:
        ppl = get_perplexity(ft_model, sent)
        results.append({"Age": col, "Perplexity": ppl, "Condition": "Finetuned"})

# Save Results
df_ppl = pd.DataFrame(results)
df_ppl.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")

# Save Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age", y="Perplexity", hue="Condition", data=df_ppl)
plt.title("Perplexity Comparison: Zeroshot vs Finetuned")
plt.xlabel("Age Group (months)")
plt.ylabel("Perplexity")
plt.legend(title="Condition")
plt.savefig(output_png)
print(f"Plot saved to {output_png}")
plt.show()