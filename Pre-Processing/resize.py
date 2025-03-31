import os
from PIL import Image

def resize_images(folder_path):
    # Target dimensions
    target_width = 1000
    target_height = 1750
    
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png']
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file is an image
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            # Full path to the image
            image_path = os.path.join(folder_path, filename)
            
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Save the resized image (overwrite original)
                    resized_img.save(image_path, quality=95)
                    print(f"Resized {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Get folder path from user
    folder_path = "c:\\Users\\Riley\\Desktop\\6Images"
    
    # Check if folder exists
    if os.path.exists(folder_path):
        resize_images(folder_path)
        print("Resizing complete!")
    else:
        print("Invalid folder path!")