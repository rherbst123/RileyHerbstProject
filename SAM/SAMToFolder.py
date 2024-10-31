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


#This script outputs each image into its own folder containig   the segmentation visualization and the cropped masks
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
        points_per_side=14,  # Number of points to sample per side of the image
        pred_iou_thresh=0.80,  # Threshold for the predicted Intersection over Union (IoU) score
        stability_score_thresh=0.80,  # Threshold for the stability score of the mask
        crop_n_layers=0,  # Number of layers to crop from the image
        crop_n_points_downscale_factor=2,  # Factor to downscale the number of points when cropping
        min_mask_region_area=4000,  # Minimum area (in pixels) for a mask region to be considered valid  # Adjusted to ignore smaller regions
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
    
    # Filter out masks that are too big
    image_area = image.shape[0] * image.shape[1]
    max_mask_area = image_area * 0.8  # Adjust this value as needed
    filtered_masks = [mask for mask in masks if mask['area'] < max_mask_area]
    
    return filtered_masks, image


# Visualize segmentation results and save to file
def visualize_and_save_segmentation(image, masks, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Correct color display

    for mask in masks:
        mask_image = mask['segmentation']
        plt.contour(mask_image, colors="red")

    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'{os.path.basename(output_folder)}_segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to crop and save each mask separately
def crop_and_save_masks(image, masks, output_folder):
    for idx, mask in enumerate(masks):
        # Get the mask
        mask_image = mask['segmentation']

        # Get the bounding box
        x, y, w, h = mask['bbox']
        x = int(max(x, 0))
        y = int(max(y, 0))
        w = int(w)
        h = int(h)

        # Ensure coordinates are within image bounds
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])

        # Crop the image using the bounding box
        cropped_image = image[y:y_end, x:x_end]

        # Apply the mask to the cropped image
        mask_cropped = mask_image[y:y_end, x:x_end]
        mask_bool = mask_cropped.astype(bool)
        for c in range(3):  # For each color channel
            cropped_image[:, :, c] = cropped_image[:, :, c] * mask_bool

        # Save the masked image
        output_file = os.path.join(output_folder, f'mask_{idx + 1}.png')
        cv2.imwrite(output_file, cropped_image)

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

                # Create a folder for this image in the output folder
                image_folder_name = os.path.splitext(image_file)[0]
                image_output_folder = os.path.join(output_folder, image_folder_name)
                os.makedirs(image_output_folder, exist_ok=True)

                with pbar_lock:
                    pbar.set_description(f'Processing {image_file}')
                    # print(f"Processing {image_file}")

                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    visualize_and_save_segmentation(image, masks, image_output_folder)
                    crop_and_save_masks(image, masks, image_output_folder)
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
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\Images"  # Update this path as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\SEGTESTINGFOLER_200ImageTest10"  # Update this path as needed
    main_pipeline(input_folder, output_folder)
