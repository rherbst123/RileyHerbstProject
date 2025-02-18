import easyocr
import numpy as np
from PIL import Image
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Folder containing the images and subfolders
base_folder = "c:\\Users\\Riley\\Desktop\SEGTESTINGFOLER_200ImageTest10-Collaged"

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

def contains_scale_bar_text(image_path):
    # Load the image and perform OCR
    img = Image.open(image_path)
    img_np = np.array(img)
    result = reader.readtext(img_np, detail=0)
    text = ' '.join(result)

    # Check for specific keywords indicating a scale bar
    scale_bar_keywords = ['cm', 'copyright', 'reserved']
    return any(keyword in text for keyword in scale_bar_keywords)

def delete_and_filter_images():
    final_collages_folder = os.path.join(base_folder, 'final collages')
    os.makedirs(final_collages_folder, exist_ok=True)

    for root, dirs, files in os.walk(base_folder):
        # Exclude 'final collages' from processing
        if 'final collages' in dirs:
            dirs.remove('final collages')

        print(f"Processing folder: {root}")
        
        kept_images = []

        for filename in files:
            file_path = os.path.join(root, filename)
            
            if 'segmentation_visualization' in filename or 'collage' in filename:
                print(f"Skipping {filename} as it's not a target image.")
                continue

            try:
                # Check for scale bar text in the image
                if contains_scale_bar_text(file_path):
                    print(f"Identified scale bar text in {filename}. Deleting...")
                    os.remove(file_path)
                    continue

                # Open the image and perform existing checks
                with Image.open(file_path) as img:
                    width, height = img.size
                    img_np = np.array(img)
                
                # Use EasyOCR to extract text and count words
                result = reader.readtext(img_np, detail=0)
                text = ' '.join(result)
                words = text.strip().split()
                word_count = len(words)
                    
                # Delete images with fewer than 5 words
                if word_count < 5:
                    print(f"Only {word_count} words detected in {filename}. Deleting...")
                    os.remove(file_path)
                    continue

                # Delete images with a resolution higher than 2000x2000
                if width > 2000 or height > 2000:
                    print(f"Image {filename} has a resolution of {width}x{height}. Deleting...")
                    os.remove(file_path)
                    continue

                # Keep the image if it passes all checks
                print(f"{word_count} words detected in {filename}. Keeping the image.")
                kept_images.append(file_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Create a collage if there are kept images
        if kept_images:
            folder_name = os.path.basename(root)
            collage_output_path = os.path.join(final_collages_folder, f"{folder_name}_collage.png")
            create_collage(kept_images, collage_output_path)

# Run the delete and filter function
delete_and_filter_images()
