import os
from paddleocr import PaddleOCR
from PIL import Image

#This Script sucks and so does paddle in this case i guess. probably me tho


# Initialize PaddleOCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Specify the directory containing the images
image_folder = "c:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER_200ImageTest4\\0000_C0000578F"

# Get a list of all .png files in the directory
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Check if the file is a valid image
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (IOError, SyntaxError):
        print(f"Skipping invalid image file: {image_file}")
        continue

    # Use PaddleOCR to detect text in the image
    result = ocr.ocr(image_path, cls=True)

    # If result is empty, no text was detected
    if not result:
        print(f"No text detected in {image_file}. Deleting the image.")
        os.remove(image_path)
    else:
        print(f"Text detected in {image_file}. Keeping the image.")

print("Processing complete.")
