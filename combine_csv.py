import csv
import os

def combine_csv_files(input_dir, output_file):
    # Get a list of all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return
    
    # Open the output file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        
        for csv_file in csv_files:
            with open(os.path.join(input_dir, csv_file), 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                headers = next(reader)  # Read the header row
                
                # Write the header to the output file only once
                if writer is None:
                    writer = csv.writer(outfile)
                    writer.writerow(headers)
                
                # Write the data rows to the output file
                for row in reader:
                    writer.writerow(row)
    
    print(f"Combined {len(csv_files)} CSV files into {output_file}.")

if __name__ == "__main__":

    output_dir = 'combined_csv_files'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define paths for the three conditions. These were generated in the previous steps and should be updated if the paths change.
    dir_list = ['zero_shot_results', 'fine_tuned_results', 'real_world_results'] 
    for directory in dir_list:
        output_filename = f'{output_dir}/combined_{directory}.csv'
        combine_csv_files(directory, output_filename)