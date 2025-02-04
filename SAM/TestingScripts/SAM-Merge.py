import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import gc
import time
import numpy as np
import easyocr
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import psutil
import GPUtil
import threading

# Function to get resource usage
def get_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_percent = gpu.load * 100
        gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
    else:
        gpu_percent, gpu_memory_percent = 0, 0
    return f"CPU: {cpu_percent:.1f}%, Mem: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%, GPU Mem: {gpu_memory_percent:.1f}%"

# Resource monitor function
def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        resource_usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(resource_usage)
        time.sleep(1)

# Function to log SAM parameters and execution time
def log_test_details(output_folder, parameters, execution_time):
    try:
        log_file = os.path.join(output_folder, "test_log.txt")
        with open(log_file, "w") as file:
            file.write("Segment Anything Model (SAM) Parameters:\n")
            for key, value in parameters.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            file.write(f"Execution Time: {execution_time:.2f} seconds\n")
        print(f"Test details logged in {log_file}")
    except Exception as e:
        print(f"Error writing log file: {e}")

# Merge overlapping masks
def merge_overlapping_masks(masks, threshold=0.2):
    def calculate_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
        inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1, area2 = w1 * h1, w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def combine_bboxes(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        x, y = min(x1, x2), min(y1, y2)
        w, h = max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y
        return x, y, w, h

    merged_masks = []
    for mask in masks:
        added = False
        for merged_mask in merged_masks:
            if calculate_iou(mask['bbox'], merged_mask['bbox']) > threshold:
                merged_mask['area'] += mask['area']
                merged_mask['bbox'] = combine_bboxes(merged_mask['bbox'], mask['bbox'])
                merged_mask['segmentation'] = np.logical_or(
                    merged_mask['segmentation'], mask['segmentation']
                ).astype(np.uint8)
                added = True
                break
        if not added:
            merged_masks.append(mask)
    return merged_masks

# Filter OCR results to exclude single-word detections
def filter_ocr_results(ocr_results, min_word_count=3):
    return [text for text in ocr_results if len(text.split()) >= min_word_count]

# Load the Segment Anything Model
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_l_0b3195.pth"  # Update path
    model_type = "vit_l"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    sam_params = {
        "points_per_side": 15,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.85,
        "crop_n_layers": 0,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 1000,
    }
    mask_generator = SamAutomaticMaskGenerator(sam, **sam_params)
    return mask_generator, sam_params

# Segment the image
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")
    scale = min(4096 / max(image.shape[:2]), 1)
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    masks = mask_generator.generate(image)
    return masks, image

# Filter masks using EasyOCR
def filter_masks(masks, image, reader):
    mask_info_list = []
    for idx, mask in enumerate(masks):
        x, y, w, h = [int(v) for v in mask['bbox']]
        cropped_image = image[y:y + h, x:x + w]
        mask_bool = mask['segmentation'][y:y + h, x:x + w].astype(bool)
        masked_image = cropped_image * mask_bool[..., None]
        resized_masked_image = cv2.resize(masked_image, (0, 0), fx=3, fy=3)

        result = reader.readtext(resized_masked_image, detail=0, paragraph=True)
        result = filter_ocr_results(result, min_word_count=3)

        if result:
            mask_info_list.append({'idx': idx, 'mask': mask, 'bbox': mask['bbox'], 'text': result})
    return mask_info_list

# Visualize and save results
def visualize_and_save_segmentation(image, mask_info_list, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for mi in mask_info_list:
        mask_image = mi['mask']['segmentation']
        plt.contour(mask_image, colors="red")
    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, "segmentation_visualization.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

# Main pipeline
def main_pipeline(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)
    reader = easyocr.Reader(['en'])
    mask_generator, sam_params = initialize_sam()

    start_time = time.time()
    with tqdm(total=len(image_files), desc="Processing Images") as pbar:
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            try:
                masks, image = generate_segmentation(image_path, mask_generator)
                masks = merge_overlapping_masks(masks, threshold=0.2)
                filtered_masks = filter_masks(masks, image, reader)

                image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
                os.makedirs(image_output_folder, exist_ok=True)
                visualize_and_save_segmentation(image, filtered_masks, image_output_folder)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
            pbar.update(1)
    execution_time = time.time() - start_time
    log_test_details(output_folder, sam_params, execution_time)

if __name__ == "__main__":
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\10Images"  # Update path
    output_folder = "C:\\Users\\Riley\\Desktop\\TestingCords7"  # Update path
    main_pipeline(input_folder, output_folder)
