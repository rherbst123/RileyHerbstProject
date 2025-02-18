import csv

def main():
    # Hardcode the input CSV file paths
    input_files = [
        r"c:\\Users\\Riley\\Desktop\\InputText\\OutputCsv\\Transcription_Claude_3.5_Sonnet_1.5Stripped_02_17_25-03_00PM.csv",
        r"c:\\Users\\Riley\\Desktop\\InputText\\OutputCsv\\Transcription_GPT-4o_1.5Stripped_02_17_25-03_01PM.csv",
        r"c:\\Users\\Riley\\Desktop\\InputText\\OutputCsv\\Transcription_Llama3.2_Vision_OllamaPrompt_02_17_25-02_53PM.csv",
   
    ]
    
    # Hardcode the output CSV file path (directory + filename)
    output_file = r"C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\CrossValidation\\Outputs\\final_results3.csv"
    
    # Read all CSVs into memory (as lists of rows),
    # trimming whitespace from each field
    all_data = []
    for file in input_files:
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = [[col.strip() for col in row] for row in reader]
            all_data.append(data)
    
    # We assume all CSVs have the same number of rows and columns
    num_rows = len(all_data[0])
    num_cols = len(all_data[0][0])
    
    # Build the final results row by row
    final_results = []
    
    for row_idx in range(num_rows):
        new_row = []
        for col_idx in range(num_cols):
            # Collect the values for this row and column across the five files
            values = [all_data[file_idx][row_idx][col_idx] for file_idx in range(3)]
            
            # Count the frequency of each distinct value
            freq = {}
            for val in values:
                freq[val] = freq.get(val, 0) + 1
            
            # Find the most common value
            most_common_value = max(freq, key=freq.get)
            most_common_count = freq[most_common_value]
            
            # Format like: "value(3/5)"
            formatted = f"{most_common_value}({most_common_count}/3)"
            new_row.append(formatted)
        
        final_results.append(new_row)
    
    # Write the final CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(final_results)

if __name__ == "__main__":
    main()
