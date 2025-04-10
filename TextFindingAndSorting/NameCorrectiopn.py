import os

directory = "c:\\Users\\Riley\\Desktop\\300ImagesCollage"
suffix_to_remove = "_collage"

for filename in os.listdir(directory):
    print(f"Processing file: {filename}")  # Debugging: Print each filename
    
    new_name = filename
    
    # Remove the suffix if it exists
    if new_name.endswith(suffix_to_remove):
        print(f"Suffix found: {suffix_to_remove}")  # Debugging: Confirm suffix match
        new_name = new_name[:-len(suffix_to_remove)]
        print(f"Removed suffix: {filename} -> {new_name}")
    
    # Rename the file if changes were made
    if new_name != filename:
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        print(f"Renaming: {filename} -> {new_name}")
        os.rename(old_path, new_path)
    else:
        print(f"No changes for: {filename}")  # Debugging: Print if no changes
