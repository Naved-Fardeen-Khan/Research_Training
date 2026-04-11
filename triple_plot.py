import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

# --- 1. CONFIGURATION ---
# Replace these with your actual file paths
paths = {
    "Zeroshot": "combined_csv_files/combined_zero_shot_results.csv",
    "Finetuned": "combined_csv_files/combined_fine_tuned_results.csv",
    "Real World": "combined_csv_files/combined_real_world_results.csv"
}
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

valid_ages = list(range(3, 79, 3)) # Months (3, 6, 9... 78)
group_size = 50  # Number of sentences to group together for TTR calculation (sliding window)
order = 2        # 2nd order polynomial for the growth curve

# Official Palette (High-Contrast, Colorblind-Friendly)
palette = {"Zeroshot": "#1b9e77", "Finetuned": "#d95f02", "Real World": "#7570b3"}

def calculate_group_ttr(text_list):
    """Calculates TTR for a concatenated block of sentences."""
    combined = " ".join(map(str, text_list)).lower()
    words = re.findall(r'\b\w+\b', combined)
    return len(set(words)) / len(words) if words else np.nan

def calculate_length(text):
    """Calculates the number of words in a text."""
    return len(re.findall(r'\b\w+\b', str(text).lower()))
   
# --- 2. DATA PROCESSING ---
all_data = []

for label, path in paths.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping {label}.")
        continue
        
    # Read CSV, ensuring we handle potentially missing headers in real-world data
    df = pd.read_csv(path)
    
    for age in valid_ages:
        col_name = f"{age} Months"
        if col_name in df.columns:
            sentences = df[col_name].dropna().tolist()
            
            # SLIDING WINDOW: This creates the "fat" violins seen in natural language
            # Instead of 1 TTR per sentence (mostly 1.0), we get 1 TTR per 10 sentences.
            for i in range(0, len(sentences), group_size):
                chunk = sentences[i : i + group_size]
                if len(chunk) >= (group_size / 2): # Allow smaller groups at the end
                    ttr_val = calculate_group_ttr(chunk)
                    length_val = calculate_length(chunk) / group_size  # Average length per sentence in the chunk
                    all_data.append({"Age": age, "TTR": ttr_val, "Condition": label, "Length": length_val})

df_final = pd.DataFrame(all_data)

# --- 3. PLOTTING TTR ---
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# A. Triple Violin Plot
# Using 'inner=None' and alpha to make the comparison cleaner
ax = sns.violinplot(data=df_final, x="Age", y="TTR", hue="Condition", 
                    palette=palette, cut=0, alpha=0.6, inner="quartile")

# B. Polynomial Trend Lines (MATLAB polyfit logic)
# We calculate the median per age to draw the dashed lines
for label in paths.keys():
    subset = df_final[df_final["Condition"] == label]
    if not subset.empty:
        # Group by age to find medians for the fit
        medians = subset.groupby("Age")["TTR"].median().reindex(valid_ages).dropna()
        
        # x_coords are the positions of the violins (0, 1, 2...)
        x_coords = [i for i, age in enumerate(valid_ages) if age in medians.index]
        y_values = medians.values
        
        # 2nd Order Polynomial Fit
        z = np.polyfit(x_coords, y_values, order)
        p = np.poly1d(z)
        
        # Plot smooth curve
        x_smooth = np.linspace(0, len(valid_ages)-1, 100)
        line_style = ":" if label == "Zeroshot" else "--" if label == "Finetuned" else "-"
        plt.plot(x_smooth, p(x_smooth), linestyle=line_style, color=palette[label], 
                 linewidth=3, label=f"{label} Trend")

# C. Formatting
plt.title("Lexical Diversity (TTR): Triple Comparison", fontsize=20, fontweight='bold')
plt.xlabel("Age of Child (Months)", fontsize=14)
plt.ylabel("Type-Token Ratio (TTR)", fontsize=14)
plt.xticks(rotation=0)
plt.ylim(.1, .75) # TTR is between 0 and 1, but we know it won't reach 1 with our grouping method
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/final_triple_comparison_plot.png', dpi=300)
print("Plot successfully generated and saved.")

# --- 4. PLOTTING LENGTH ---
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# A. Triple Violin Plot for Length
ax = sns.violinplot(data=df_final, x="Age", y="Length", hue="Condition", 
                    palette=palette, cut=0, alpha=0.6, inner="quartile")
# B. Polynomial Trend Lines for Length
for label in paths.keys():
    subset = df_final[df_final["Condition"] == label]
    if not subset.empty:
        medians = subset.groupby("Age")["Length"].median().reindex(valid_ages).dropna()
        x_coords = [i for i, age in enumerate(valid_ages) if age in medians.index]
        y_values = medians.values
        z = np.polyfit(x_coords, y_values, order)
        p = np.poly1d(z)
        line_style = ":" if label == "Zeroshot" else "--" if label == "Finetuned" else "-"
        plt.plot(x_smooth, p(x_smooth), linestyle=line_style, color=palette[label], 
                 linewidth=3, label=f"{label} Length Trend")
# C. Formatting
plt.title("Average Utterance Length: Triple Comparison", fontsize=20, fontweight='bold')
plt.xlabel("Age of Child (Months)", fontsize=14)
plt.ylabel("Average Utterance Length (Words)", fontsize=14)
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/final_length_triple_comparison_plot.png', dpi=300)
print("Length plot successfully generated and saved.")