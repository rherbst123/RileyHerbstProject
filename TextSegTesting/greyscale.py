from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\riley\\Desktop\\Code\\PythonForWork\\images\\0009_V0075836F.jpg"
image = Image.open(image_path)

# Convert the image to greyscale
greyscale_image = image.convert('L')

# Convert back to RGB to apply saturation
greyscale_image_rgb = greyscale_image.convert('RGB')

# Increase the saturation
enhancer = ImageEnhance.Color(greyscale_image_rgb)
saturated_image = enhancer.enhance(2)  # Increase saturation by a factor of 2

# Save the result
saturated_image.save('C:\\Users\\riley\\Desktop\\Code\\PythonForWork\\images\\saturated_image.jpg')

# Display the images
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image)
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title('Greyscale Image')
# plt.imshow(greyscale_image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Saturated Image')
# plt.imshow(saturated_image)
# plt.axis('off')

# plt.show()