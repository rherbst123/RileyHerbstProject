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
from torchvision.ops import boxes

# Patch for boxes.batched_nms
_original_batched_nms = boxes.batched_nms
def patched_batched_nms(boxes_tensor, scores, idxs, iou_threshold):
    boxes_tensor = boxes_tensor.cpu()
    scores = scores.cpu()
    idxs = idxs.cpu()
    return _original_batched_nms(boxes_tensor, scores, idxs, iou_threshold)
boxes.batched_nms = patched_batched_nms

def get_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
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

def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        resource_usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(resource_usage)
        time.sleep(1)

# Initialize the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "C:\\Users\\riley\\Desktop\\sam_vit_h_4b8939.pth"  
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=19,          # Number of points per side of the image
        pred_iou_thresh=0.90,        # IoU threshold for predictions
        stability_score_thresh=0.92, # Stability threshold
        crop_n_layers=1,             # Layers to crop
        crop_n_points_downscale_factor=0.7, # Downscale factor for points during crop
        min_mask_region_area=14500,  # Minimum area for valid mask region
    )

# Generate segmentation masks and resized image if necessary
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")
    max_dimension = 2250
    scale = max_dimension / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    masks = mask_generator.generate(image)
    
    # Filter out masks that are overly large compared to image area
    image_area = image.shape[0] * image.shape[1]
    max_mask_area = image_area * 0.8
    filtered_masks = [mask for mask in masks if mask['area'] < max_mask_area]
    
    return filtered_masks, image

# Compute Intersection over Union for two masks
def compute_mask_iou(mask1, mask2):
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    iou = intersection / (union + 1e-6)
    return iou

# Remove duplicate masks based on IoU threshold
def remove_duplicate_masks(masks, iou_threshold=0.80):
    unique_masks = []
    for mask in masks:
        duplicate = False
        for unique in unique_masks:
            iou = compute_mask_iou(mask['segmentation'], unique['segmentation'])
            if iou > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            unique_masks.append(mask)
    return unique_masks

# Erode the binary mask to create a separation (buffer) between segments
def erode_mask(mask, kernel_size=3, iterations=1):
    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)
    return (eroded_mask > 0)

# Visualize segmentation results and save as an image file
def visualize_and_save_segmentation(image, masks, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for mask in masks:
        mask_image = mask['segmentation']
        plt.contour(mask_image, colors="red")
    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'{os.path.basename(output_folder)}_segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# Crop each mask from the image after applying erosion to create a separation
def crop_and_save_masks(image, masks, output_folder, erosion_kernel_size=3, erosion_iterations=1):
    for idx, mask in enumerate(masks):
        # Apply erosion on the mask segmentation
        mask_image = mask['segmentation']
        eroded_mask = erode_mask(mask_image, kernel_size=erosion_kernel_size, iterations=erosion_iterations)
        
        # Get bounding box coordinates
        x, y, w, h = mask['bbox']
        x = int(max(x, 0))
        y = int(max(y, 0))
        w = int(w)
        h = int(h)
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        
        # Crop image and corresponding mask region
        cropped_image = image[y:y_end, x:x_end]
        mask_cropped = eroded_mask[y:y_end, x:x_end]
        mask_bool = mask_cropped.astype(bool)
        for c in range(3):  # Apply the mask to each channel
            cropped_image[:, :, c] = cropped_image[:, :, c] * mask_bool

        output_file = os.path.join(output_folder, f'mask_{idx + 1}.png')
        cv2.imwrite(output_file, cropped_image)
        print(f"Segment {idx + 1}: Coordinates (x: {x}, y: {y}, width: {w}, height: {h}), File: {output_file}")

# Main pipeline for processing a folder of images
def main_pipeline(input_folder, output_folder):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    num_images = len(image_files)
    if num_images == 0:
        print(f"No images found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)
    pbar_lock = threading.Lock()
    with tqdm(total=num_images, desc='Processing Images', unit='image') as pbar:
        # Start resource monitor thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            with pbar_lock:
                pbar.set_description('Initializing SAM')
            mask_generator = initialize_sam()

            for image_file in image_files:
                image_path = os.path.join(input_folder, image_file)
                
                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    # Remove duplicate masks based on IoU
                    masks = remove_duplicate_masks(masks, iou_threshold=0.80)
                    image_folder_name = os.path.splitext(image_file)[0]
                    image_output_folder = os.path.join(output_folder, image_folder_name)
                    os.makedirs(image_output_folder, exist_ok=True)

                    with pbar_lock:
                        pbar.set_description(f'Processing {image_file}')
                    
                    visualize_and_save_segmentation(image, masks, image_output_folder)
                    crop_and_save_masks(image, masks, image_output_folder, erosion_kernel_size=3, erosion_iterations=1)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                with pbar_lock:
                    pbar.update(1)

            # Clean up the SAM model after processing
            del mask_generator
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            # Stop the resource monitor thread
            stop_event.set()
            monitor_thread.join()

if __name__ == "__main__":
    # Check versions and GPU availability
    if torch.cuda.is_available():
        print("Cuda Version:", torch.version.cuda)
        print("GPU Used:", torch.cuda.get_device_name(0))
        print("Current GPU Code Used:", torch.cuda.current_device())
        print("Number of GPUs installed:", torch.cuda.device_count())
    else:
        print("No GPU available")

    print("Starting...")
    
    input_folder = "c:\\Users\\Riley\\Desktop\\300ImageTess\\300Images_4_14_25_SatBri_Completed"  # Update as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\300ImagesSegmentted4_15_25_FourthRun"               # Update as needed
    main_pipeline(input_folder, output_folder)
