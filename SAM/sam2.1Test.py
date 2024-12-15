import torch
import sam2_configs
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling import sam2_base
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


# Load the SAM 2.1 model
def initialize_sam2():
    sam2_checkpoint = "c:\\Users\\Riley\\Desktop\\sam2.1_hiera_base_plus.pt"  # Update with your SAM 2.1 checkpoint path
    model_cfg = "c:\\Users\\Riley\\Desktop\\sam2.1_hiera_b+.yaml"  # Update this path as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


# Segment the image using SAM 2.1
def generate_segmentation(image_path, predictor):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_dimension = 4096
    scale = max_dimension / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    predictor.set_image(image)
    input_point = np.array([[500, 375]])  # Example point; replace with dynamic input
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

    return masks, image


# Filter masks using EasyOCR (same as provided earlier)
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


# Visualize segmentation results (same as provided earlier)
def visualize_and_save_segmentation(image, masks, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask in masks:
        plt.contour(mask.squeeze(), colors="red")

    plt.axis("off")
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'{os.path.basename(output_folder)}_segmentation_visualization.png')
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()


# Main pipeline
def main_pipeline(input_folder, output_folder):
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    num_images = len(image_files)

    if num_images == 0:
        print(f"No images found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)

    reader = easyocr.Reader(["en"])
    pbar_lock = threading.Lock()
    with tqdm(total=num_images, desc="Processing Images", unit="image") as pbar:
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor_thread.start()

        try:
            with pbar_lock:
                pbar.set_description("Initializing SAM 2.1")
            predictor = initialize_sam2()

            for image_file in image_files:
                image_path = os.path.join(input_folder, image_file)

                image_folder_name = os.path.splitext(image_file)[0]
                image_output_folder = os.path.join(output_folder, image_folder_name)
                os.makedirs(image_output_folder, exist_ok=True)

                with pbar_lock:
                    pbar.set_description(f"Processing {image_file}")

                try:
                    masks, image = generate_segmentation(image_path, predictor)
                    visualize_and_save_segmentation(image, masks, image_output_folder)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

                with pbar_lock:
                    pbar.update(1)

            del predictor
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            stop_event.set()
            monitor_thread.join()


if __name__ == "__main__":
    input_folder = "C:\\Users\\Riley\\Desktop\\Portal\\Code\\Images"  # Update this path as needed
    output_folder = "C:\\Users\\Riley\\Desktop\\BigTest"  # Update this path as needed
    main_pipeline(input_folder, output_folder)
