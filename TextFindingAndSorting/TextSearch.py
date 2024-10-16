import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as needed

# Folder containing the images and subfolders
base_folder = "c:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER_200ImageTest2"

# Iterate through each folder and subfolder
# for root, dirs, files in os.walk(base_folder):
#     for filename in files:
#         file_path = os.path.join(root, filename)
        
#         # Check if the file is the segmentation_visualization and skip deletion
#         if 'segmentation_visualization' in filename:
#             print(f"Skipping {filename} as it's the segmentation_visualization.")
#             continue
        
#         try:
#             # Open the image file
#             with Image.open(file_path) as img:
#                 # Check the image size
#                 width, height = img.size
                
#                 # If the resolution is higher than 1000x1000, delete the image
#                 if width > 12000 or height > 1900:
#                     img.close()  # Ensure the file is closed before deleting
#                     print(f"Image {filename} has a resolution of {width}x{height}. Deleting...")
#                     os.remove(file_path)
#                     continue
                
#                 # Use pytesseract to extract text
#                 text = pytesseract.image_to_string(img)
                
#                 # If no text is detected, delete the image
#                 if not text.strip():
#                     img.close()  # Ensure the file is closed before deleting
#                     print(f"No text detected in {filename}. Deleting...")
#                     os.remove(file_path)
#                 else:
#                     print(f"Text detected in {filename}. Keeping the image.")
                    
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")



for root, dirs, files in os.walk(base_folder):
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
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(img)
                
                # Split the text into words
                words = text.strip().split()
                word_count = len(words)
                
                # If fewer than 7 words are detected, delete the image
                if word_count < 7:
                    print(f"Only {word_count} words detected in {filename}. Deleting...")
                    os.remove(file_path)
                    continue

                # If the resolution is higher than 2000x2000, delete the image
                if width > 2000 or height > 2000:
                    print(f"Image {filename} has a resolution of {width}x{height}. Deleting...")
                    os.remove(file_path)
                    continue

                else:
                    print(f"{word_count} words detected in {filename}. Keeping the image.")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")