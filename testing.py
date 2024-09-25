import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from paddleocr import PaddleOCR
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# Initialize BLIP2 Model for captioning
def initialize_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate an overall caption
def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\SAM\\sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)

# Segment the image
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    masks = mask_generator.generate(image)
    return masks, image

# Visualize segmentation results
def visualize_segmentation(image, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask in masks:
        mask_image = mask['segmentation']
        plt.contour(mask_image, colors=[np.random.rand(3,)])
    
    plt.show()

# OCR for text detection (optional)
def detect_text(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path, cls=True)
    return result

# Putting it all together
def main_pipeline(image_path):
    # Step 1: Captioning using BLIP2
    processor, model = initialize_blip()
    caption = generate_caption(image_path, processor, model)
    #print(f"Overall Caption: {caption}")

    # Step 2: Segmentation using SAM
    mask_generator = initialize_sam()
    masks, image = generate_segmentation(image_path, mask_generator)
    visualize_segmentation(image, masks)

    # Step 3: Region-based captioning (manual, using bounding boxes)
    # This would involve integrating GRiT, which isn't directly available on Hugging Face yet

    # Step 4: Optional Text Detection using OCR
    text_detections = detect_text(image_path)
    print(f"Text Detections: {text_detections}")

if __name__ == "__main__":
    # Sample image path
    image_path = "c:\\Users\\Riley\\Documents\\Github\\RileyHerbstProject\\PaddleOCR\\V0577013F.jpg"
    main_pipeline(image_path)
