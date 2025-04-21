import os
import base64
import requests
import time

API_KEY = "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"
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
USER_PROMPT_FILE   = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\Prompt_1.5.3.txt"

TEXT_FOLDER = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4_Confirmed_Gpto4\Finished"      # folder with .txt inputs
OUTPUT_FILE = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4_Confirmed_Gpto4\FinalOutput\GPTo4Traanscribed_Gpto4Confirmed.txt"


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_text_files(folder: str):
    return sorted(
        fn for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn)) and fn.lower().endswith(".txt")
    )

system_prompt = read_text(SYSTEM_PROMPT_FILE)
user_prompt   = read_text(USER_PROMPT_FILE)
text_files    = get_text_files(TEXT_FOLDER)

print(f"Found {len(text_files)} text files.")

separator = "=" * 50 + "\n"
headers   = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

start_time = time.time()

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for idx, filename in enumerate(text_files, 1):
        file_path   = os.path.join(TEXT_FOLDER, filename)
        file_text   = read_text(file_path)

        # Combine your static user prompt with the file’s content
        user_message = f"{user_prompt}\n\n{file_text}"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens": 4098,
            "temperature": 0.0,
            "seed": 42,
        }

        print(f"Processing file {idx}: '{filename}'")
        response      = requests.post("https://api.openai.com/v1/chat/completions",
                                      headers=headers, json=payload)
        response_data = response.json()
        assistant_reply = (
            response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if response.ok else f"Error {response.status_code}: {response.text}"
        )

        # Write result
        outfile.write(f"Entry {idx}\nFile: {filename}\n{assistant_reply}\n")
        outfile.write(separator)

        print(f"File {idx} processed.")

elapsed = time.time() - start_time
print(f"All files processed in {elapsed:.2f} seconds.")
print(f"Results saved to {OUTPUT_FILE}")