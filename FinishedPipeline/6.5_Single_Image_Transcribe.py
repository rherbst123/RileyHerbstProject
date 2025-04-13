import os
import base64
import requests
import time

API_KEY = "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"
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT_FILE = "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\SystemPrompt.txt"
USER_PROMPT_FILE = "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\Prompt_1.5.2.txt"
IMAGE_FOLDER = "c:\\Users\\Riley\\Desktop\\300ImagesCollage"  
OUTPUT_FILE = "c:\\Users\\Riley\\Desktop\\300Images_4_10_25_Only_Collaage_GPT4o.txt"

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
image_files = get_image_files(IMAGE_FOLDER)
print(f"Found {len(image_files)} images.")

separator = "=" * 50 + "\n"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

start_time = time.time()

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for idx, img_filename in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGE_FOLDER, img_filename)
        encoded_image = encode_image_to_base64(image_path)
        image_type = determine_type(image_path)
        message_content = [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_type};base64,{encoded_image}"
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
        print(f"Processing image {idx}: '{img_filename}'")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            assistant_reply = response_data["choices"][0].get("message", {}).get("content", "")
        else:
            assistant_reply = "No valid response returned from API."
        entry_text = (
            f"Entry {idx}\n"
            f"Image: {img_filename}\n"
            f"{assistant_reply}\n"
        )
        outfile.write(entry_text)
        outfile.write(separator)
        print(f"Image {idx} processed.")

elapsed_time = time.time() - start_time
print(f"All images processed in {elapsed_time:.2f} seconds.")
print(f"Results saved to {OUTPUT_FILE}")
