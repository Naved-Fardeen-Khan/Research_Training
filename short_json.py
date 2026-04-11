import os
import json
import re
# Create shorter JSON file from combined.jsonl for quicker testing
input_file = 'combined.jsonl'
print(f'Total lines in {input_file}: ', sum(1 for line in open(input_file, 'r', encoding='utf-8')))
output_file = 'shorter.jsonl'
max_entries = 100
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        if i >= max_entries:
            break
        last_line = line
        outfile.write(line)
print(f"Created {output_file} with up to {max_entries} entries from {input_file}.")
print(f'Last line in {output_file}: ', last_line)
