import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm
import psutil
import GPUtil
import threading
import time

# Function to get resource usage
def get_resource_usage():
    # Get CPU usage
    cpu_percent = psutil.cpu_percent()
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    # Get GPU usage
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_percent = gpu.load * 100
        gpu_memory_used = gpu.memoryUsed
        gpu_memory_total = gpu.memoryTotal
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
    else:
        gpu_percent = 0
        gpu_memory_percent = 0
    return f"CPU:{cpu_percent:.1f}%, Mem:{memory_percent:.1f}%, GPU:{gpu_percent:.1f}%, GPU Mem:{gpu_memory_percent:.1f}%"

# Resource monitor function
def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        resource_usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(resource_usage)
        time.sleep(1)

# Initialize BLIP Model for captioning
def initialize_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.half()  # Use FP16
    return processor, model

# Generate an overall caption
def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(dtype=torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_h_4b8939.pth"  # Use the 'vit_h' checkpoint
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,  # Reduced from 32
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,  # Increased to ignore smaller regions
    )

# Segment the image
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    max_dimension = 1024
    scale = max_dimension / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    masks = mask_generator.generate(image)
    return masks, image

# Visualize segmentation results
def visualize_segmentation(image, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Correct color display

    for mask in masks:
        mask_image = mask['segmentation']
        plt.contour(mask_image, colors=[np.random.rand(3,)])

    plt.show()

# OCR for text detection (optional)
def detect_text(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    result = ocr.ocr(image_path, cls=True)
    return result

# Putting it all together with TQDM and resource monitoring
def main_pipeline(image_path):
    steps = ['Captioning', 'Segmentation', 'Text Detection']
    pbar_lock = threading.Lock()
    with tqdm(total=len(steps), desc='Processing', unit='step') as pbar:
        # Start resource monitor thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            # Step 1: Captioning using BLIP
            with pbar_lock:
                pbar.set_description('Initializing BLIP')
            processor, model = initialize_blip()
            with pbar_lock:
                pbar.set_description('Generating Caption')
            caption = generate_caption(image_path, processor, model)
            print(f"Overall Caption: {caption}")
            del model  # Free up memory
            torch.cuda.empty_cache()
            gc.collect()
            with pbar_lock:
                pbar.update(1)

            # Step 2: Segmentation using SAM
            with pbar_lock:
                pbar.set_description('Initializing SAM')
            mask_generator = initialize_sam()
            with pbar_lock:
                pbar.set_description('Generating Segmentation')
            masks, image = generate_segmentation(image_path, mask_generator)
            visualize_segmentation(image, masks)
            del mask_generator
            torch.cuda.empty_cache()
            gc.collect()
            with pbar_lock:
                pbar.update(1)

            # Step 3: Optional Text Detection using OCR
            with pbar_lock:
                pbar.set_description('Detecting Text')
            text_detections = detect_text(image_path)
            print(f"Text Detections: {text_detections}")
            with pbar_lock:
                pbar.update(1)
        finally:
            # Stop the resource monitor thread
            stop_event.set()
            monitor_thread.join()

if __name__ == "__main__":
    # Sample image path
    print("Starting...")
    image_path = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\Images\\0022_K000536956.jpg"
    main_pipeline(image_path)
