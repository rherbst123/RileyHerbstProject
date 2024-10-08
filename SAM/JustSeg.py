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
        points_per_side=16,  # Adjusted as needed
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,  # Adjusted to ignore smaller regions
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

# Main pipeline processing a folder of images
def main_pipeline(input_folder, output_folder):
    # Get list of image files in the input folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
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
                    #print(f"Processing {image_file}")

                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    visualize_and_save_segmentation(image, masks, output_path)
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
    #Versions checking
    

    if torch.cuda.is_available():
        print("Cuda Version:",torch.version.cuda)
        print("GPU Used:",torch.cuda.get_device_name(0))
        print("Current Gpu Code Used:",torch.cuda.current_device())
        print("Number of GPU's installed:",torch.cuda.device_count())
    else:
        print("No GPU available")

    print("Starting...")
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\10Images"  # Update this path as needed
    output_folder = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\SegImages"  # Update this path as needed
    main_pipeline(input_folder, output_folder)
