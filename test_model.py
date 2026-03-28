import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import matplotlib.pyplot as plt
import re
import os

def ttr_calculate(text):
# Combine all sentences and clean punctuation
    all_text = " ".join(text).lower()
    tokens = re.findall(r'\b\w+\b', all_text)
    
    if not tokens:
        return 0
    
    types = set(tokens)
    return len(types) / len(tokens)

# --- 1. SETTINGS ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-lora/final_adapter"

# VARIABLES 
age_in_months = list(range(3, 85, 3))  # 3, 6, 9, ..., 84 months
topics = ["food", "toys", "animals", "family", "dog", "cat", "house", "car", "bird"]
num_generations = 50
ttr_list = []
all_speeches = []

# --- 2. LOAD MODEL & ADAPTER ---
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)

# Load LoRA adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

for age in age_in_months:
    # Output parameters:
    temperature = 0.8 # Higher temperature means more randomness in the output
    top_p = 0.9       # Top-p sampling means the model will only consider the smallest set of tokens whose cumulative probability exceeds top_p.

    current_age_speeches = []
    print(f"\nUsing parameters: temperature={temperature}, top_p={top_p}\n")

    for i in range(num_generations):

        topic = random.choice(topics)
    # --- 3. GENERATION LOOP ---
    # We use the same formatting as the training data
        messages = [
            {"role": "user", "content": f"Generate a sentence of child-directed speech for a {age}-month-old infant about {topic}."},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs.input_ids.shape[1]
        #print(f"\n--- Generating {num_generations} speeches for a {age}-month-old ---")
        #print(f"\nPrompt: {messages[0]['content']}")
        with torch.no_grad():
            # Output settings:
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=temperature,
                do_sample=True,    # Must be True to get different results each time
                top_p=top_p,        
                pad_token_id=tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        new_tokens = outputs[0][prompt_length:]
        speech = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        #print(f"{i+1}. {speech}")
        current_age_speeches.append(speech)

        # Optional: Add a short delay between generations to avoid overwhelming the output
        # time.sleep(0.5)

    # Calculate TTR for the current age group
    ttr = ttr_calculate(current_age_speeches)
    ttr_list.append(ttr)
    all_speeches.append(current_age_speeches)


print("\nAll generations completed.")

# Create a directory for results if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# --- 4. PLOT TTR RESULTS ---
plt.figure(figsize=(10, 6))
plt.plot(age_in_months, ttr_list, marker='o', linestyle='-', color='b', linewidth=2)

plt.title('Lexical Diversity (TTR) vs. Child Age')
plt.xlabel('Age (Months)')
plt.ylabel('Type-Token Ratio (TTR)')
plt.xticks(age_in_months)   
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot as a file (standard for Puhti)
while True:
    i = 1
    try:
        plt.savefig(f'results/ttr_diversity_plot{i}.png')
        break
    except FileExistsError:
        i += 1

print(f'TTR plot saved as results/ttr_diversity_plot{i}.png')


# --- 5. SAVE ALL SPEECHES IN A 2D MATRIX STRUCTURE ---
import csv

# all_speeches is currently: [[age3_s1, age3_s2...], [age6_s1, age6_s2...]]
# zip(*all_speeches) turns it into: [(age3_s1, age6_s1...), (age3_s2, age6_s2...)]
while True:
    i = 1
    try:
        with open(f'results/speeches_matrix{i}.csv', 'x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Write the Header (Ages as columns)
            header = [f"{age} Months" for age in age_in_months]
            writer.writerow(header)
            
            # Write the Rows (Sentences)
            # zip(*all_speeches) pairs the 1st sentence of every age, then the 2nd, etc.
            for row in zip(*all_speeches):
                writer.writerow(row)

        print(f"Matrix saved as 'results/speeches_matrix{i}.csv'")
        break
    except FileExistsError:
        i += 1