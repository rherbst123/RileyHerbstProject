import easyocr
import numpy as np
from PIL import Image
import os

# Create an EasyOCR reader (specify the languages you want to read)
reader = easyocr.Reader(['en'])  # You can add more language codes as needed

# Folder containing the images and subfolders
base_folder = "C:\\Users\\riley\\Desktop\\TestingCords2"

# Iterate over each file in the base folder
for root, dirs, files in os.walk(base_folder):
    print(f"Processing folder: {root}")  # Print the current folder being processed
    for filename in files:
        file_path = os.path.join(root, filename)
        
        # Check if the file is the segmentation_visualization and skip deletion
        if 'segmentation_visualization' in filename:
            print(f"Skipping {filename} as it's the segmentation_visualization.")
            continue

        try:
            # Open the image file
            with Image.open(file_path) as img:
                width, height = img.size
                img_np = np.array(img)
            
            # Use EasyOCR to extract text
            result = reader.readtext(img_np, detail=0)
            text = ' '.join(result)
                
            # Split the text into words
            words = text.strip().split()
            word_count = len(words)
                
            if word_count < 4:
                print(f"Only {word_count} words detected in {filename}. Deleting...")
                os.remove(file_path)
                continue

            # If the resolution is higher than 2000x2000, delete the image
            if width > 2000 or height > 2000:
                print(f"Image {filename} has a resolution of {width}x{height}. Deleting...")
                os.remove(file_path)
                continue

            # Check if the word 'copyrighted' is in the detected text
            if 'copyrighted' in text.lower():
                print(f"The word 'copyrighted' detected in {filename}. Deleting...")
                os.remove(file_path)
                continue

            else:
                print(f"{word_count} words detected in {filename}. Keeping the image.")
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
