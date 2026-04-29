import os
import random
import re
import csv
import matplotlib.pyplot as plt

def ttr_calculate(text):
    # Combine all sentences and clean punctuation
    all_text = " ".join(text).lower()
    tokens = re.findall(r'\b\w+\b', all_text)
    
    if not tokens:
        return 0
    
    types = set(tokens)
    return len(types) / len(tokens)

# --- SETTINGS ---
path = 'txt_data' 
target_ages = list(range(3, 85, 3)) # The same bins used for generation
samples_per_age = 100

real_world_ttr_results = []
all_real_speeches = []

# Regex to find the folders
age_pattern = re.compile(r'childes_age_(\d+)')

print("Sampling real-world data for TTR comparison...")

for target_age in target_ages:
    if target_age < 10:
        folder_name = f"childes_age_00{target_age}"
        full_folder_path = os.path.join(path, folder_name)
    else:
        folder_name = f"childes_age_0{target_age}"
        full_folder_path = os.path.join(path, folder_name)
    
    age_pool = []
    
    if os.path.exists(full_folder_path):
        # 1. Collect ALL possible lines for this age
        for txt_file in os.listdir(full_folder_path):
            with open(os.path.join(full_folder_path, txt_file), 'r', encoding='utf-8') as f:
                for line in f:
                    clean = line.strip()
                    if clean:
                        age_pool.append(clean)
        
        # 2. Randomly sample 100
        if len(age_pool) >= samples_per_age:
            sample = random.sample(age_pool, samples_per_age)
        else:
            # Fallback if a folder is small
            sample = age_pool 
            print(f"Warning: Only found {len(age_pool)} lines for age {target_age}")
            
        # 3. Calculate TTR for the REAL data
        real_ttr = ttr_calculate(sample) 
        real_world_ttr_results.append(real_ttr)
        all_real_speeches.append(sample)
        
        print(f"Age {target_age} Real TTR: {real_ttr:.4f}")
    else:
        print(f"Folder not found for age {target_age}, skipping...")
        real_world_ttr_results.append(None)

print("Real-world sampling complete.")

# Create a directory for real-world results if it doesn't exist
if not os.path.exists('real_world_results'):
    os.makedirs('real_world_results')

# Save the real-world TTR results as png and speeches as csv

# 1. Plot TTR results
plt.figure(figsize=(10, 6))
plt.plot(target_ages, real_world_ttr_results, marker='o', linestyle='-', color='g', linewidth=2)
plt.title('Real-World Lexical Diversity (TTR) vs. Child Age')
plt.xlabel('Age (Months)')
plt.ylabel('Type-Token Ratio (TTR)')
plt.xticks(target_ages)
plt.grid(True, linestyle='--', alpha=0.7)

# 2. Save the plot
i = 1
while True:
    filename = f'real_world_results/real_world_ttr_plot{i}.png'
    if not os.path.exists(filename):
        plt.savefig(filename)
        break
    i += 1 
print(f'Real-world TTR plot saved as real_world_results/real_world_ttr_plot{i}.png')

# 3. Save the speeches in a CSV
with open(f'real_world_results/real_world_speeches_matrix{i}.csv', 'x', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # Write the Header (Ages as columns)
    header = [f"{age} Months" for age in target_ages]
    writer.writerow(header)
    
    # Write the rows (zip to align speeches by index across ages)
    for row in zip(*all_real_speeches):
        writer.writerow(row)

print(f"Real-world speeches saved as 'real_world_results/real_world_speeches_matrix{i}.csv'")