from PIL import Image, ImageEnhance

def adjust_image(image_path, output_path, brightness_factor=1.0, saturation_factor=1.0):

    # Open the original image
    image = Image.open(image_path)

    # 1) Adjust brightness
    brightness_enhancer = ImageEnhance.Brightness(image)
    image_adjusted_brightness = brightness_enhancer.enhance(brightness_factor)

    # 2) Adjust saturation
    saturation_enhancer = ImageEnhance.Color(image_adjusted_brightness)
    final_image = saturation_enhancer.enhance(saturation_factor)

    # Save and return final image
    final_image.save(output_path)
    return final_image

if __name__ == "__main__":
    # Example usage:
    # Adjust the brightness and saturation of "input.jpg" 
    # Brightness = 1.2 (20% brighter), Saturation = 1.5 (50% more saturated)
    # Then display the resulting image.
    
    input_image_path = "c:\\Users\\Riley\\Desktop\\TestSet\\0010_BR0000006918378.jpg"
    output_image_path = "c:\\Users\\Riley\\Desktop\\output.jpg"
    
    processed_img = adjust_image(
        image_path=input_image_path,
        output_path=output_image_path,
        brightness_factor=0.7,
        saturation_factor=2.0
    )
    
    # Show the final image
    processed_img.show()
