import csv
import re

def parse_txt_file(input_filename):
    with open(input_filename, "r", encoding="utf-8") as f:
        content = f.read()

    
    entries = re.split(r"\n=+\n", content.strip())
    data = []

    for entry in entries:
        lines = entry.splitlines()
        record = {}
        if not lines:
            continue
       
        if lines[0].strip().startswith("Entry"):
            record["Entry"] = lines[0].strip()
            fields = lines[1:]
        else:
            fields = lines

        # Process each line that contains a colon.
        for line in fields:
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
        if record:
            data.append(record)
    return data

def get_fieldnames_in_order(data):
   
    fieldnames = []
    for record in data:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames

def write_to_csv(data, output_filename):
    
    fieldnames = get_fieldnames_in_order(data)
    
    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data:
            writer.writerow(record)

def main():
    input_filename = "FinishedPipeline/Outputs/TXT/Gpt260_ClaudeConfirmed.txt"  # The input text file
    output_filename = "FinishedPipeline/Outputs/CSV/Gpt260_ClaudeConfirmed.csv"   # The output CSV file

    # Parse the text file into structured data
    data = parse_txt_file(input_filename)
    # Write the data to CSV with field order preserved
    write_to_csv(data, output_filename)
    print(f"Conversion complete. CSV file created: {output_filename}")

if __name__ == "__main__":
    main()
