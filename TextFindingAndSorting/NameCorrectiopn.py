import os

directory = "c:\\Users\\Riley\\Desktop\\260Images_4_14_25_Extract_Then_Parse"
prefix_to_remove = "sat_"

for foldername in os.listdir(directory):
    folder_path = os.path.join(directory, foldername)
    
    # Check if the current item is a folder
    if os.path.isdir(folder_path):
        print(f"Processing folder: {foldername}")  # Debugging: Print each folder name
        
        new_name = foldername
        
        # Remove the prefix if it exists
        if new_name.startswith(prefix_to_remove):
            print(f"Prefix found: {prefix_to_remove}")  # Debugging: Confirm prefix match
            new_name = new_name[len(prefix_to_remove):]
            print(f"Removed prefix: {foldername} -> {new_name}")
        
        # Rename the folder if changes were made
        if new_name != foldername:
            old_path = folder_path
            new_path = os.path.join(directory, new_name)
            print(f"Renaming: {foldername} -> {new_name}")
            os.rename(old_path, new_path)
        else:
            print(f"No changes for: {foldername}")  # Debugging: Print if no changes
