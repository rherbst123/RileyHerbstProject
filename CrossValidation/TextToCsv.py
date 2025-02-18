import os
import csv
import re

def parse_local_images(text):
    """
    Given the full text of a single .txt file, split it into blocks
    that start with 'Local Image' or are separated by lines of '=' chars.
    Then parse each block's lines as key-value pairs.
    Returns a list of dictionaries: one for each 'Local Image'.
    """
    # Split on the "Local Image" lines for a rough segmentation
    # You might also want to split on '===' or combine both approaches.
    # This example tries to be forgiving by matching lines that start with "Local Image".
    blocks = re.split(r'(?:^|\n)(Local Image.*?)(?=\nLocal Image|\n?$)', text, flags=re.DOTALL)

    records = []
    current_record = {}

    # Because of how re.split with capturing groups works, blocks can come in pairs:
    #   [ '', 'Local Image 1\nFilename: ...', 'Some text', 'Local Image 2\nFilename: ...', '...' ]
    # We will merge each "heading" piece with the piece that follows it to get a single "block".
    def parse_block(block):
        """
        Parse lines in a block for 'key: value' format and return a dict.
        """
        record = {}
        lines = block.splitlines()
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Attempt to match lines like "key: value"
            if ':' in line:
                # Split once at the first colon (some data might contain extra colons)
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                record[key] = val
        return record

    # Combine the heading with the next block to parse them together
    # so we don’t lose the “Local Image X” or “Filename: ...” lines.
    # The first piece could be empty if the file starts with "Local Image..."
    i = 0
    while i < len(blocks):
        block_heading = blocks[i].strip()
        # If there is another chunk, combine them. Otherwise parse just the one.
        if i + 1 < len(blocks):
            block_content = block_heading + "\n" + blocks[i+1]
            i += 2
        else:
            block_content = block_heading
            i += 1

        # Filter out truly empty text
        if block_content.strip():
            record = parse_block(block_content)
            # Only add to list if there's meaningful data
            if record:
                records.append(record)

    return records

def write_csv(records, output_csv):
    """
    Given a list of dictionaries (records) and a path to an output CSV,
    write all records as rows in the CSV. Use union of all keys to form columns.
    """
    # Gather all field names from the union of all record keys
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())
    fieldnames = sorted(all_keys)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

def process_txt_file(input_path, output_path):
    """
    Parse a single text file and write its corresponding CSV to output_path.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Parse local images from the text
    records = parse_local_images(text)
    # Write out CSV
    write_csv(records, output_path)


def main(input_folder, output_folder):
    """
    For every .txt file in 'input_folder', parse it and produce
    a matching CSV in 'output_folder'.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .txt file
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            # Construct CSV filename by replacing '.txt' with '.csv'
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_folder, csv_filename)

            print(f"Processing {filename} -> {csv_filename}")
            process_txt_file(input_path, output_path)


if __name__ == "__main__":
    # Example usage:
    # 1) Create an 'input_texts' folder containing the .txt files
    # 2) Create an 'output_csvs' folder for the CSVs
    # 3) Run: python parse_texts_to_csv.py

    input_folder = "C:\\Users\\Riley\\Desktop\\InputText"   # Adjust as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\InputText\\OutputCsv"  # Adjust as needed
    main(input_folder, output_folder)
