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

# VARIABLES 
age_in_months = list(range(3, 85, 3))  # 3, 6, 9, ..., 84 months
topics = ["food", "toys", "animals", "family", "dog", "cat", "house", "car", "bird"]
num_generations = 100
batch_size = 20
ttr_list = []
all_speeches = []

# --- 2. LOAD MODEL & ADAPTER ---

# Load just the base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

for age in age_in_months:
    # Output parameters:
    temperature = 0.8 # Higher temperature means more randomness in the output
    top_p = 0.9       # Top-p sampling means the model will only consider the smallest set of tokens whose cumulative probability exceeds top_p.

    current_age_speeches = []
    print(f"\n--- Using parameters: temperature={temperature}, top_p={top_p} ---")
    print(f"\n--- Processing Age {age} months (5 topics x 20 batch size) ---")

    for i in range(num_generations//batch_size):

        topic = random.choice(topics)
    # --- 3. GENERATION LOOP ---
    # We use the same formatting as the training data
        messages = [
            {"role": "user", "content": f"Generate a sentence of child-directed speech for a {age}-month-old infant about {topic}."},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs.input_ids.shape[1]     
        #print(f"\nPrompt: {messages[0]['content']}")

        with torch.no_grad():
            # Output settings:
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=temperature,
                do_sample=True,    # Must be True to get different results each time
                top_p=top_p,
                num_return_sequences=batch_size,        
                pad_token_id=tokenizer.eos_token_id
            )

            for output in outputs:
                speech = tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
                #print(f"{i+1}. {speech}")
                current_age_speeches.append(speech)

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        new_tokens = outputs[0][prompt_length:]
        speech = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"{i+1}. {speech}")

        # Optional: Add a short delay between generations to avoid overwhelming the output
        # time.sleep(0.5)

    # Calculate TTR for the current age group
    ttr = ttr_calculate(current_age_speeches)
    ttr_list.append(ttr)
    all_speeches.append(current_age_speeches)


print("\nAll generations completed.")

# Create a directory for zero-shot results if it doesn't exist
if not os.path.exists('zero_shot_results'):
    os.makedirs('zero_shot_results')

# --- 4. PLOT TTR RESULTS ---
plt.figure(figsize=(10, 6))
plt.plot(age_in_months, ttr_list, marker='o', linestyle='-', color='b', linewidth=2)

plt.title('Lexical Diversity (TTR) vs. Child Age for 100 Generations per Age Group')
plt.xlabel('Age (Months)')
plt.ylabel('Type-Token Ratio (TTR)')
plt.xticks(age_in_months)   
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot as a file (standard for Puhti)
i = 1
while True:
    filename = f'zero_shot_results/ttr_diversity_plot{i}.png'
    if not os.path.exists(filename):
        plt.savefig(filename)
        break
    i += 1

print(f'TTR plot saved as zero_shot_results/ttr_diversity_plot{i}.png')


# --- 5. SAVE ALL SPEECHES IN A 2D MATRIX STRUCTURE ---
import csv

# all_speeches is currently: [[age3_s1, age3_s2...], [age6_s1, age6_s2...]]
# zip(*all_speeches) turns it into: [(age3_s1, age6_s1...), (age3_s2, age6_s2...)]
with open(f'zero_shot_results/speeches_matrix{i}_with_temp{temperature}_and_top_p{top_p}.csv', 'x', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
            
    # Write the Header (Ages as columns)
    header = [f"{age} Months" for age in age_in_months]
    writer.writerow(header)
            
    # Write the Rows (Sentences)
    # zip(*all_speeches) pairs the 1st sentence of every age, then the 2nd, etc.
    for row in zip(*all_speeches):
        writer.writerow(row)

print(f"Matrix saved as 'zero_shot_results/speeches_matrix{i}_with_temp{temperature}_and_top_p{top_p}.csv'")
        