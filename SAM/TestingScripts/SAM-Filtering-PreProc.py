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
from skimage import exposure  # For gamma adjustment

# Function to get resource usage
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

# Resource monitor function
def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        resource_usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(resource_usage)
        time.sleep(1)

# Preprocess the image: grayscale, boost saturation, and adjust gamma
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to BGR (to retain 3 channels for further processing)
    grayscale_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # Boost saturation
    hsv_image = cv2.cvtColor(grayscale_bgr, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], 50)  # Increase saturation
    boosted_saturation_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Lower gamma
    adjusted_gamma_image = exposure.adjust_gamma(boosted_saturation_image, gamma=0.75)  # Lower gamma

    return adjusted_gamma_image

# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_l_0b3195.pth"  # Update this path as needed
    model_type = "vit_l"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=15,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.80,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
    )

# Segment the image with preprocessing
def generate_segmentation(image_path, mask_generator):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    max_dimension = 4096
    scale = max_dimension / max(preprocessed_image.shape[:2])
    if scale < 1:
        preprocessed_image = cv2.resize(
            preprocessed_image, (int(preprocessed_image.shape[1] * scale), int(preprocessed_image.shape[0] * scale))
        )

    masks = mask_generator.generate(preprocessed_image)

    # Filter out masks that are too big
    image_area = preprocessed_image.shape[0] * preprocessed_image.shape[1]
    max_mask_area = image_area * 0.9  # Allow slightly larger masks
    filtered_masks = [mask for mask in masks if mask["area"] < max_mask_area]

    return filtered_masks, preprocessed_image

# Initialize EasyOCR
def initialize_easyocr():
    return easyocr.Reader(['en'])

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

        cropped_image = image[y:y_end, x:x_end]

        mask_image = mask['segmentation']
        mask_cropped = mask_image[y:y_end, x:x_end]
        mask_bool = mask_cropped.astype(bool)
        masked_image = np.zeros_like(cropped_image)
        for c in range(3):
            masked_image[:, :, c] = cropped_image[:, :, c] * mask_bool

        if masked_image.shape[0] < 32 or masked_image.shape[1] < 32:
            continue

        scale_factor = 2
        resized_masked_image = cv2.resize(masked_image, (0, 0), fx=scale_factor, fy=scale_factor)

        result = reader.readtext(resized_masked_image, detail=0, paragraph=False)
        num_words = sum([len(text.split()) for text in result])

        mask_info['num_words'] = num_words
        mask_info['contains_text'] = num_words >= 1
        mask_info_list.append(mask_info)

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

# Visualize segmentation results
def visualize_and_save_segmentation(image, mask_info_list, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for mi in mask_info_list:
        mask_image = mi['mask']['segmentation']
        plt.contour(mask_image, colors="red")

    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'{os.path.basename(output_folder)}_segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# Main pipeline processing a folder of images
def main_pipeline(input_folder, output_folder):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    num_images = len(image_files)

    if num_images == 0:
        print(f"No images found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)
    reader = initialize_easyocr()

    pbar_lock = threading.Lock()
    with tqdm(total=num_images, desc='Processing Images', unit='image') as pbar:
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            with pbar_lock:
                pbar.set_description('Initializing SAM')
            mask_generator = initialize_sam()

            for image_file in image_files:
                image_path = os.path.join(input_folder, image_file)
                image_folder_name = os.path.splitext(image_file)[0]
                image_output_folder = os.path.join(output_folder, image_folder_name)
                os.makedirs(image_output_folder, exist_ok=True)

                with pbar_lock:
                    pbar.set_description(f'Processing {image_file}')

                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    filtered_mask_info_list = filter_masks(masks, image, reader)
                    visualize_and_save_segmentation(image, filtered_mask_info_list, image_output_folder)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

                with pbar_lock:
                    pbar.update(1)

            del mask_generator
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            stop_event.set()
            monitor_thread.join()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Cuda Version:", torch.version.cuda)
        print("GPU Used:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\Images"
    output_folder = "C:\\Users\\Riley\\Desktop\\TestingCords7"
    main_pipeline(input_folder, output_folder)
