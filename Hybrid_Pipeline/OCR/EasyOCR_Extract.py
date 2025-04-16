import os
import shutil
import easyocr

# Define the base folder that holds the subfolders with .png images.
base_folder = 'C:\\Users\\Riley\\Desktop\\300ImagesSegmentted4_14_25_ThirdRun_OCRd'

# Initialize the EasyOCR reader for English (set gpu=True if you have a supported GPU).
reader = easyocr.Reader(['en'])

# Iterate over the items inside the base folder.
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    
    # Process only directories.
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")
        
        # Define paths for the new images and rawText folders inside this subfolder.
        images_folder = os.path.join(subfolder_path, 'images')
        rawtext_folder = os.path.join(subfolder_path, 'rawText')
        
        # Create these destination folders if they don't exist.
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(rawtext_folder, exist_ok=True)
        
        # List all files in the subfolder.
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            
            # Skip directories (such as the ones we just created).
            if os.path.isdir(file_path):
                continue

            # Process only .png images.
            if file.lower().endswith('.png'):
                print(f"  Processing image: {file_path}")
                
                # Copy the image into the images_folder.
                destination_img_path = os.path.join(images_folder, file)
                shutil.copy(file_path, destination_img_path)
                
                # Run EasyOCR on the copied image (or the original, if preferred).
                results = reader.readtext(file_path, detail=0)  # detail=0 returns only text strings
                extracted_text = "\n".join(results)
                
                # Define the text file name with the same base name.
                base_name = os.path.splitext(file)[0]
                txt_filename = f"{base_name}.txt"
                txt_path = os.path.join(rawtext_folder, txt_filename)
                
                # Write the extracted text to the .txt file.
                with open(txt_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)
                
                print(f"    Text written to: {txt_path}")

print("Processing complete.")