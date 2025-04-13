import os
import base64
import requests
import time

API_KEY = "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"
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT_FILE = "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\SystemPrompt.txt"
USER_PROMPT_FILE = "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\Prompt_1.5.2.txt"
IMAGE_FOLDER_1 = "c:\\Users\\Riley\\Desktop\\300ImagesFull"
IMAGE_FOLDER_2 = "c:\\Users\\Riley\\Desktop\\300ImagesCollage"
OUTPUT_FILE = "c:\\Users\\Riley\\Desktop\\300Images_4_9_25.txt"

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def determine_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    else:
        return "image/jpeg"

def get_image_files(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png")
    files = [filename for filename in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(valid_extensions)]
    return sorted(files)

system_prompt = read_text_file(SYSTEM_PROMPT_FILE)
user_prompt = read_text_file(USER_PROMPT_FILE)
folder1_images = get_image_files(IMAGE_FOLDER_1)
folder2_images = get_image_files(IMAGE_FOLDER_2)

if len(folder1_images) != len(folder2_images):
    print("Warning: The two folders have a different number of images. Proceeding with the minimum count.")
pair_count = min(len(folder1_images), len(folder2_images))
print(f"Found {pair_count} paired images.")

separator = "=" * 50 + "\n"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

start_time = time.time()

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for idx, (img1_filename, img2_filename) in enumerate(zip(folder1_images, folder2_images), start=1):
        image1_path = os.path.join(IMAGE_FOLDER_1, img1_filename)
        image2_path = os.path.join(IMAGE_FOLDER_2, img2_filename)
        encoded_image1 = encode_image_to_base64(image1_path)
        encoded_image2 = encode_image_to_base64(image2_path)
        type1 = determine_type(image1_path)
        type2 = determine_type(image2_path)
        message_content = [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{type1};base64,{encoded_image1}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{type2};base64,{encoded_image2}"
                }
            }
        ]
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message_content
            }
        ]
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 4098,
            "temperature": 0.0,
            "seed": 42
        }
        print(f"Processing pair {idx}: '{img1_filename}' and '{img2_filename}'")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            assistant_reply = response_data["choices"][0].get("message", {}).get("content", "")
        else:
            assistant_reply = "No valid response returned from API."
        entry_text = (
            f"Entry {idx}\n"
            f"Images: {img1_filename}, {img2_filename}\n"
            f"{assistant_reply}\n"
        )
        outfile.write(entry_text)
        outfile.write(separator)
        print(f"Pair {idx} processed.")

elapsed_time = time.time() - start_time
print(f"All pairs processed in {elapsed_time:.2f} seconds.")
print(f"Results saved to {OUTPUT_FILE}")
