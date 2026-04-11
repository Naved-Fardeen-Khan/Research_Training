import os # For file and directory operations
import json # For JSON handling
import re # For regular expressions
import random # For random selection of instruction templates


root = '/Users/fardeen/Documents/Study/Research_Training/txt_data' # Root directory containing age group folders

if not os.path.exists(root):
    raise FileNotFoundError(f"The specified root directory '{root}' does not exist.")

# Maybe for future use.
'''
mother_json = 'mot.jsonl' # Output JSON file for mother utterances
father_json = 'fat.jsonl' # Output JSON file for father utterances
'''

output_file = '/Users/fardeen/Documents/Study/Research_Training/combined.jsonl' # Output JSON file for combined mother and father utterances

# Regex pattern to extract age from folder name of format 'childes_age_<age in months>'
age_pattern = re.compile(r'childes_age_(\d+)')

# Varied instruction templates to prevent overfitting to a single prompt
instruction_templates = [
    "Generate child-directed speech from a parent to a {age}-month-old child.",     # 1
    "You are a parent. What would you say to your {age}-month-old baby?",           # 2
    "Speak to a {age}-month-old infant as a father or mother would.",               # 3
    "Give an example of baby-talk a parent would use with a {age}-month-old.",      # 4
    "Imagine you're talking to a {age}-month-old child. What would you say?",       # 5
    "Create a sentence that a parent might say to their {age}-month-old baby.",     # 6
    "What would a mother or father say to a {age}-month-old child?",                # 7
    "Generate a sentence of child-directed speech for a {age}-month-old infant.",   # 8
    "As a parent, how would you talk to your {age}-month-old baby?",                # 9
    "Write a sentence that a parent might say to their {age}-month-old child."      # 10
]

print("Starting conversion to JSONL...")

with open(output_file, 'w', encoding='utf-8') as out_file:
    for age_folder in sorted(os.listdir(root)):
        age_match = age_pattern.match(age_folder)
        if not age_match: # Skip folder that don't match the age pattern(unknown age)
            continue
        print(f"Processing folder: {age_folder}")
        age_in_months = int(age_match.group(1))

        for txt_file in os.listdir(os.path.join(root, age_folder)):
            with open(os.path.join(root, age_folder, txt_file), 'r', encoding='utf-8') as file:
                for line in file:
                    clean_line = line.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').strip() # Clean up spacing around punctuation

                    if not clean_line.strip(): # Skip empty lines
                        continue

                    # Randomly select an instruction template and format it with the age
                    instruction = random.choice(instruction_templates).format(age=age_in_months)

                    entry = {
                        "instruction": instruction,
                        "output": clean_line
                    }
                    json.dump(entry, out_file, ensure_ascii=False)
                    out_file.write('\n')

number_of_entries = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
print(f"Conversion to JSON completed. Number of entries: {number_of_entries}")
        