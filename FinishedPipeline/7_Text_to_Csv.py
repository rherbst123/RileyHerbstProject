import csv
import re

def parse_txt_file(input_filename):
    """Reads the text file and returns a list of dictionaries (records),
    each preserving the insertion order of keys."""
    with open(input_filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content by delimiter lines (assumes a line with only '=' characters)
    entries = re.split(r"\n=+\n", content.strip())
    data = []

    for entry in entries:
        lines = entry.splitlines()
        record = {}
        if not lines:
            continue
        # If the first line starts with "Entry", store it as a separate field.
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
    """Collects field names in the order of first occurrence across the data."""
    fieldnames = []
    for record in data:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames

def write_to_csv(data, output_filename):
    """Writes the list of dictionaries into a CSV file while preserving field order."""
    # Get field names preserving the order of first appearance.
    fieldnames = get_fieldnames_in_order(data)
    
    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data:
            writer.writerow(record)

def main():
    input_filename = "FinishedPipeline/Outputs/TXT/300Images_4_10_25_Only_Collaage_GPT4o.txt"  # The input text file
    output_filename = "FinishedPipeline/Outputs/CSV/300Images_4_10_25_Only_Collaage_GPT4o.csv"   # The output CSV file

    # Parse the text file into structured data
    data = parse_txt_file(input_filename)
    # Write the data to CSV with field order preserved
    write_to_csv(data, output_filename)
    print(f"Conversion complete. CSV file created: {output_filename}")

if __name__ == "__main__":
    main()
