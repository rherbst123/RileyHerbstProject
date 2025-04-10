import os
from PIL import Image, ImageEnhance

def adjust_image(image_path, output_path, brightness_factor, saturation_factor):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)

    # Apply brightness adjustment
    brightness_enhancer = ImageEnhance.Brightness(image)
    image_adjusted_brightness = brightness_enhancer.enhance(brightness_factor)
    print(f"Brightness adjusted by factor: {brightness_factor}")

    # Apply saturation adjustment
    saturation_enhancer = ImageEnhance.Color(image_adjusted_brightness)
    final_image = saturation_enhancer.enhance(saturation_factor)
    print(f"Saturation adjusted by factor: {saturation_factor}")

    # Save the final image
    final_image.save(output_path)
    print(f"Image saved to: {output_path}")

def process_folder(input_folder, output_folder, brightness_factor, saturation_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"sat_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            adjust_image(input_path, output_path, brightness_factor, saturation_factor)

if __name__ == "__main__":
    input_folder = "c:\\Users\\Riley\\Desktop\\300Images(4_9_25)"
    output_folder = "c:\\Users\\Riley\\Desktop\\300Images(4_9_25)-SatBri-Completed"

    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        brightness_factor=0.9,
        saturation_factor=1.2
    )

    print("Processing complete. Images saved to:", output_folder)
