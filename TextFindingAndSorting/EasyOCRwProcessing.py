import easyocr
import numpy as np
from PIL import Image, ImageOps
import os
import math

# Create an EasyOCR reader (specify the languages you want to read)
reader = easyocr.Reader(['en'])  # You can add more language codes as needed

# Folder containing the images and subfolders
base_folder = "c:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER_200ImageTest7-BlackBackground"

#This Dont work!!!!!!!!!

def create_neat_collage(image_paths, output_path, grid_size=(5, 5), image_size=(200, 200), padding=10, background_color=(0, 0, 0)):
    """
    Creates a collage with images arranged in a neat grid.

    :param image_paths: List of image file paths.
    :param output_path: File path to save the collage image.
    :param grid_size: Tuple indicating the number of images in (columns, rows).
    :param image_size: Tuple indicating the size to which each image will be resized.
    :param padding: Space between images in pixels.
    :param background_color: Background color of the collage.
    """
    cols, rows = grid_size
    thumb_width, thumb_height = image_size
    images = []

    # Load and resize images
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)  # Updated
            images.append(img)
        except Exception as e:
            print(f"Error processing image {image_path} for collage: {e}")
            continue

    if not images:
        print("No images to create collage.")
        return

    # Calculate collage dimensions
    collage_width = cols * thumb_width + (cols + 1) * padding
    collage_height = rows * thumb_height + (rows + 1) * padding

    # Create the collage image
    collage_image = Image.new('RGB', (collage_width, collage_height), color=background_color)

    # Paste images into the collage
    index = 0
    for row in range(rows):
        for col in range(cols):
            if index >= len(images):
                break
            x = padding + col * (thumb_width + padding)
            y = padding + row * (thumb_height + padding)
            collage_image.paste(images[index], (x, y))
            index += 1

    collage_image.save(output_path)
    print(f"Collage saved to {output_path}")

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
                
            if word_count < 5:
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
        collage_output_path = os.path.join(root, f'collage_{len(kept_images)}.png')
        # Adjust grid size based on the number of images
        num_images = len(kept_images)
        cols = min(5, num_images)  # Maximum 5 columns
        rows = math.ceil(num_images / cols)
        create_neat_collage(kept_images, collage_output_path, grid_size=(cols, rows))

