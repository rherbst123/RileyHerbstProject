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


# Function to get resource usage
def get_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_percent = gpu.load * 100
        gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
    else:
        gpu_percent, gpu_memory_percent = 0, 0
    return f"CPU:{cpu_percent:.1f}%, Mem:{memory:.1f}%, GPU:{gpu_percent:.1f}%, GPU Mem:{gpu_memory_percent:.1f}%"


# Resource monitor function
def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        resource_usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(resource_usage)
        time.sleep(1)


# Initialize SAM
def initialize_sam():
    sam_checkpoint = "c:\\Users\\Riley\\Desktop\\sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=2500,
    )


# Edge detection and density calculation
def calculate_edge_density(mask, image):
    x, y, w, h = mask['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, threshold1=50, threshold2=150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = roi.shape[0] * roi.shape[1]
    return edge_pixels / total_pixels if total_pixels > 0 else 0


# Initialize EasyOCR
def initialize_easyocr():
    return easyocr.Reader(['en'])


# Generate segmentation masks
def generate_segmentation(image_path, mask_generator):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")
    max_dimension = 4096
    scale = max_dimension / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    masks = mask_generator.generate(image)
    image_area = image.shape[0] * image.shape[1]
    max_mask_area = image_area * 0.9
    return [mask for mask in masks if mask['area'] < max_mask_area], image


# Filter masks using edge density and OCR
def filter_masks_with_edges(masks, image, reader, edge_density_thresh=0.05):
    mask_info_list = []
    image_area = image.shape[0] * image.shape[1]

    for idx, mask in enumerate(masks):
        if mask['area'] > image_area * 0.9 or mask['area'] < image_area * 0.001:
            continue
        edge_density = calculate_edge_density(mask, image)
        if edge_density < edge_density_thresh:
            continue
        x, y, w, h = map(int, mask['bbox'])
        cropped_image = image[y:y+h, x:x+w]
        result = reader.readtext(cropped_image, detail=0, paragraph=False)
        num_words = sum(len(text.split()) for text in result)
        mask_info_list.append({
            'idx': idx,
            'mask': mask,
            'area': mask['area'],
            'bbox': mask['bbox'],
            'edge_density': edge_density,
            'num_words': num_words,
            'contains_text': num_words >= 5
        })
    return [mi for mi in mask_info_list if mi['contains_text']]


# Visualize and save segmentation
def visualize_and_save_segmentation(image, mask_info_list, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for mi in mask_info_list:
        plt.contour(mi['mask']['segmentation'], colors="red")
    plt.axis('off')
    plt.tight_layout()
    output_file = os.path.join(output_folder, 'segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


# Crop and save masks
def crop_and_save_masks(image, mask_info_list, output_folder):
    for idx, mi in enumerate(mask_info_list):
        x, y, w, h = map(int, mi['bbox'])
        x_end, y_end = min(x + w, image.shape[1]), min(y + h, image.shape[0])
        cropped_image = image[y:y_end, x:x_end]
        mask = mi['mask']['segmentation']
        mask_cropped = mask[y:y_end, x:x_end]
        mask_bool = mask_cropped.astype(bool)
        for c in range(3):
            cropped_image[:, :, c] *= mask_bool
        output_file = os.path.join(output_folder, f'mask_{idx + 1}.png')
        cv2.imwrite(output_file, cropped_image)


# Main pipeline
def main_pipeline(input_folder, output_folder):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"No images found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)
    reader = initialize_easyocr()

    pbar_lock = threading.Lock()
    with tqdm(total=len(image_files), desc='Processing Images', unit='image') as pbar:
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            with pbar_lock:
                pbar.set_description('Initializing SAM')
            mask_generator = initialize_sam()

            for image_file in image_files:
                image_path = os.path.join(input_folder, image_file)
                image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
                os.makedirs(image_output_folder, exist_ok=True)
                with pbar_lock:
                    pbar.set_description(f'Processing {image_file}')
                try:
                    masks, image = generate_segmentation(image_path, mask_generator)
                    filtered_masks = filter_masks_with_edges(masks, image, reader)
                    visualize_and_save_segmentation(image, filtered_masks, image_output_folder)
                    crop_and_save_masks(image, filtered_masks, image_output_folder)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                with pbar_lock:
                    pbar.update(1)
            del mask_generator
        finally:
            stop_event.set()
            monitor_thread.join()


if __name__ == "__main__":
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\Images"
    output_folder = "C:\\Users\\Riley\\Desktop\\BigTest1"
    main_pipeline(input_folder, output_folder)
