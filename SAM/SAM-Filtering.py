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
import easyocr  # EasyOCR for text detection
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

# Function to log SAM parameters and execution time
# Function to log SAM parameters and execution time
def log_test_details(output_folder, parameters, execution_time):
    try:
        log_file = os.path.join(output_folder, "test_log.txt")
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
        with open(log_file, "a") as file:  # Use append mode to avoid overwriting
            file.write("Segment Anything Model (SAM) Parameters:\n")
            for key, value in parameters.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            file.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        print(f"Test details successfully logged in {log_file}")
    except Exception as e:
        print(f"Error writing log file: {e}")



# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_l_0b3195.pth"  # Update this path as needed
    model_type = "vit_l"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=18,  # Increased from 10 to 14
        pred_iou_thresh=0.86,  # Slightly lowered to allow more masks
        stability_score_thresh=0.86,  # Slightly lowered
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1100,  # Lowered from 4000 to 1000 to include smaller regions
    )

def process_quadrants(image, mask_generator):
    height, width = image.shape[:2]
    quadrants = [
        image[:height//2, :width//2],   # Top-left
        image[:height//2, width//2:],  # Top-right
        image[height//2:, :width//2],  # Bottom-left
        image[height//2:, width//2:]   # Bottom-right
    ]
    masks = []
    for quadrant in quadrants:
        masks.extend(mask_generator.generate(quadrant))
    return masks


# Initialize EasyOCR
def initialize_easyocr():
    return easyocr.Reader(['en'])  # You can specify other languages if needed

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
    max_mask_area = image_area * 0.9  # Allow slightly larger masks
    filtered_masks = [mask for mask in masks if mask['area'] < max_mask_area]

    return filtered_masks, image

# Filter masks using EasyOCR
def filter_masks(masks, image, reader):
    mask_info_list = []
    for idx, mask in enumerate(masks):
        mask_info = {}
        mask_info['idx'] = idx
        mask_info['mask'] = mask
        mask_info['area'] = mask['area']
        mask_info['bbox'] = mask['bbox']

        x, y, w, h = mask['bbox']
        x = int(max(x, 0))
        y = int(max(y, 0))
        w = int(w)
        h = int(h)
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])

        # Crop the image using the bounding box
        cropped_image = image[y:y_end, x:x_end]

        # Apply the mask to the cropped image
        mask_image = mask['segmentation']
        mask_cropped = mask_image[y:y_end, x:x_end]
        mask_bool = mask_cropped.astype(bool)
        masked_image = np.zeros_like(cropped_image)
        for c in range(3):  # For each color channel
            masked_image[:, :, c] = cropped_image[:, :, c] * mask_bool

        # Resize the masked image for better OCR detection
        if masked_image.shape[0] < 32 or masked_image.shape[1] < 32:
            continue  # Skip too small images
        scale_factor = 2  # Increase size for better OCR accuracy
        resized_masked_image = cv2.resize(masked_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # Use EasyOCR to detect text in the masked image
        result = reader.readtext(resized_masked_image, detail=0, paragraph=False)
        num_words = sum([len(text.split()) for text in result])

        mask_info['num_words'] = num_words
        mask_info['contains_text'] = num_words >= 5  # Lowered threshold to include more text regions
        mask_info_list.append(mask_info)

    # Remove larger segments that contain smaller text segments
    indices_to_remove = set()
    for mi_text in mask_info_list:
        if not mi_text['contains_text']:
            continue
        x1, y1, w1, h1 = mi_text['bbox']
        x1_end = x1 + w1
        y1_end = y1 + h1
        area1 = mi_text['area']
        idx1 = mi_text['idx']
        for mi in mask_info_list:
            idx2 = mi['idx']
            if idx2 == idx1:
                continue
            area2 = mi['area']
            if area2 <= area1:
                continue
            x2, y2, w2, h2 = mi['bbox']
            x2_end = x2 + w2
            y2_end = y2 + h2
            # Check if the larger mask contains the smaller text mask
            if x2 <= x1 and y2 <= y1 and x2_end >= x1_end and y2_end >= y1_end:
                indices_to_remove.add(idx2)

    filtered_mask_info_list = []
    for mi in mask_info_list:
        idx = mi['idx']
        if idx in indices_to_remove:
            continue
        if not mi['contains_text']:
            continue
        filtered_mask_info_list.append(mi)

    return filtered_mask_info_list

# Visualize segmentation results and save to file
def visualize_and_save_segmentation(image, mask_info_list, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Correct color display

    for mi in mask_info_list:
        mask_image = mi['mask']['segmentation']
        plt.contour(mask_image, colors="red")

    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'{os.path.basename(output_folder)}_segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to crop and save each mask separately
def crop_and_save_masks(image, mask_info_list, output_folder):
    for idx, mi in enumerate(mask_info_list):
        mask = mi['mask']
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

        # Print coordinates of each segment and their file name
        print(f"Segment {idx + 1}: Coordinates (x: {x}, y: {y}, width: {w}, height: {h}), File: {output_file}")

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

    # Initialize EasyOCR reader
    reader = initialize_easyocr()

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
                    # Filter masks
                    filtered_mask_info_list = filter_masks(masks, image, reader)
                    visualize_and_save_segmentation(image, filtered_mask_info_list, image_output_folder)
                    crop_and_save_masks(image, filtered_mask_info_list, image_output_folder)
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
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\10Images"  # Update this path as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\FilteringTest9"  # Update this path as needed
    main_pipeline(input_folder, output_folder)
