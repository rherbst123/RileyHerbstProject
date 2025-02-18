import easyocr
import numpy as np
from PIL import Image
import os

# Create an EasyOCR reader (specify the languages you want to read)
reader = easyocr.Reader(['en'])  # You can add more language codes as needed

# Folder containing the images and subfolders
base_folder = "c:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER_200ImageTest10-finer"

def create_collage(image_paths, output_path, max_width=2000, background_color=(0, 0, 0)):
    images = []

    # Load all images
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error processing image {image_path} for collage: {e}")
            continue

    if not images:
        print("No images to create collage.")
        return

    # Sort images by height (optional)
    images.sort(key=lambda img: img.size[1])

    # Initialize variables for arranging images
    rows = []
    current_row = []
    current_width = 0
    max_row_height = 0

    for img in images:
        img_width, img_height = img.size
        if current_width + img_width <= max_width:
            current_row.append(img)
            current_width += img_width
            max_row_height = max(max_row_height, img_height)
        else:
            rows.append((current_row, max_row_height))
            current_row = [img]
            current_width = img_width
            max_row_height = img_height

    if current_row:
        rows.append((current_row, max_row_height))

    # Calculate collage dimensions
    collage_width = max_width
    collage_height = sum(height for (_, height) in rows)

    # Create the collage image
    collage_image = Image.new('RGB', (collage_width, collage_height), color=background_color)

    # Paste images into the collage
    y_offset = 0
    for row_images, row_height in rows:
        x_offset = 0
        for img in row_images:
            collage_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]
        y_offset += row_height

    collage_image = collage_image.crop(collage_image.getbbox())  # Crop to content
    collage_image.save(output_path)
    print(f"Collage saved to {output_path}")

# Function to print words detected in each image and check for numeric percentage
def print_and_check_words(words, filename):
    print(f"Words detected in {filename}:")
    num_numeric_words = 0
    for word in words:
        print(f"  - {word}")
        if word.isnumeric():
            num_numeric_words += 1

    # Check if 90% or more words are numbers
    if len(words) > 0 and (num_numeric_words / len(words)) >= 0.8:
        print(f"Image {filename} has 90% or more numbers. Deleting...")
        return True  # Signal to delete the image
    return False

# Iterate over each file in the base folder
for root, dirs, files in os.walk(base_folder):
    print(f"Processing folder: {root}")  # Print the current folder being processed

    kept_images = []  # List to store paths of images that are kept

    for filename in files:
        file_path = os.path.join(root, filename)
        
        # Check if the file is the segmentation_visualization or the collage itself, and skip it
        if 'segmentation_visualization' in filename or 'collage' in filename:
            print(f"Skipping {filename} as it's not a target image.")
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

            # Print out the words and check if image has 90% numbers
            if print_and_check_words(words, filename):
                os.remove(file_path)
                continue
            
            if word_count < 3:
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
                kept_images.append(file_path)
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # After processing all files in the current folder, create a collage if there are kept images
    if kept_images:
        collage_output_path = os.path.join(root, 'collage.png')
        create_collage(kept_images, collage_output_path)
