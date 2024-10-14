import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm
import psutil
import GPUtil
import threading
import time
import os

# As of 10.8.24 I am using Python 3.12.7 with CUDA 11.8 and Torch 2.4.1.
# Results may vary or may not work depending on your packages.

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

# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_h_4b8939.pth"  # Update this path as needed
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,  # Number of points to sample per side of the image
        pred_iou_thresh=0.90,  # Threshold for the predicted Intersection over Union (IoU) score
        stability_score_thresh=0.95,  # Threshold for the stability score of the mask
        crop_n_layers=0,  # Number of layers to crop from the image
        crop_n_points_downscale_factor=2,  # Factor to downscale the number of points when cropping
        min_mask_region_area=5500,  # Minimum area (in pixels) for a mask region to be considered valid  # Adjusted to ignore smaller regions
    )

# Segment the image
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")
    max_dimension = 4096
    scale = max_dimension / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    masks = mask_generator.generate(image)
    return masks, image

# Visualize segmentation results and save to file
def visualize_and_save_segmentation(image, masks, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Correct color display

    for mask in masks:
        mask_image = mask['segmentation']
        plt.contour(mask_image, colors="black")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# New function to crop and collage the three largest masks
def crop_and_collage_largest_masks(image, masks, output_base_path):
    # Sort masks by area in descending order
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Get the top 3 masks (or fewer if less are available)
    #TEST AROUND WITH THIS AND SEE WHAT WORKS BEST. BUT SEE IF YOU CAN DO MASKS THAT ONLY CONTAIN SOMETHING INSIDE OF THEM.
    #SOME MASKS ARE MADE AND ARE JUST A DIFFERENCE IN COLOR SO ITS NOT REALLY DOING ANYTHING.
    top_masks = sorted_masks[:2]

    cropped_images = []

    for idx, mask in enumerate(top_masks):
        # Get the bounding box
        x, y, w, h = mask['bbox']
        x = int(max(x, 0))
        y = int(max(y, 0))
        w = int(w)
        h = int(h)

        # Ensure coordinates are within image bounds
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])

        # Crop the image
        cropped_image = image[y:y_end, x:x_end]
        cropped_images.append(cropped_image)

    if not cropped_images:
        print("No masks to collage.")
        return

    # Resize images to have the same height
    min_height = min(img.shape[0] for img in cropped_images)
    resized_images = []
    for img in cropped_images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(aspect_ratio * min_height)
        resized_img = cv2.resize(img, (new_width, min_height))
        resized_images.append(resized_img)

   
    collage = cv2.hconcat(resized_images)

    #Save Segmented Portions.
    base_name, ext = os.path.splitext(output_base_path)
    output_path = f"{base_name}_segmented{ext}"
    cv2.imwrite(output_path, collage)

# Main pipeline processing a folder of images
def main_pipeline(input_folder, output_folder):
    # Get list of image files in the input folder
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    num_images = len(image_files)

    if num_images == 0:
        print(f"No images found in {input_folder}.")
        return

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    pbar_lock = threading.Lock()
    with tqdm(total=num_images, desc='Processing Images', unit='image') as pbar:
        # Start resource monitor thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            # Initialize SAM once
            with pbar_lock:
                pbar.set_description('Initializing SAM')
            mask_generator = initialize_sam()

            for image_file in image_files:
                image_path = os.path.join(input_folder, image_file)
                output_path = os.path.join(output_folder, image_file)

                with pbar_lock:
                    pbar.set_description(f'Processing {image_file}')
                    # print(f"Processing {image_file}")

                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    visualize_and_save_segmentation(image, masks, output_path)
                    crop_and_collage_largest_masks(image, masks, output_path)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                finally:
                    # Clean up to free memory
                    torch.cuda.empty_cache()
                    gc.collect()

                with pbar_lock:
                    pbar.update(1)

            # Clean up SAM model after processing
            del mask_generator
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            # Stop the resource monitor thread
            stop_event.set()
            monitor_thread.join()

if __name__ == "__main__":
    # Versions checking

    if torch.cuda.is_available():
        print("Cuda Version:", torch.version.cuda)
        print("GPU Used:", torch.cuda.get_device_name(0))
        print("Current GPU Code Used:", torch.cuda.current_device())
        print("Number of GPUs installed:", torch.cuda.device_count())
    else:
        print("No GPU available")

    print("Starting...")
    input_folder = "c:\\Users\\Riley\\Desktop\\TestSet"  # Update this path as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER"  # Update this path as needed
    main_pipeline(input_folder, output_folder)
