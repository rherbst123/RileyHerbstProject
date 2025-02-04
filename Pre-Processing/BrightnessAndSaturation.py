from PIL import Image, ImageEnhance

def adjust_image(image_path, output_path, brightness_factor, saturation_factor):

    image = Image.open(image_path)

    brightness_enhancer = ImageEnhance.Brightness(image)
    image_adjusted_brightness = brightness_enhancer.enhance(brightness_factor)

    saturation_enhancer = ImageEnhance.Color(image_adjusted_brightness)
    final_image = saturation_enhancer.enhance(saturation_factor)

    final_image.save(output_path)
    return final_image

if __name__ == "__main__":
    
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
