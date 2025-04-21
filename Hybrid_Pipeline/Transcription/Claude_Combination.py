import os
import requests
import time

# ——— CONFIG ———
API_KEY             = "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"  # ← your Claude API key here
MODEL_NAME          = "claude-3-7-sonnet-20250219"  # or "claude-3.7-sonnet-20250219"
SYSTEM_PROMPT_FILE  = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
USER_PROMPT_FILE    = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\Prompt_1.5.3.txt"

TEXT_FOLDER         = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4_Confirmed_Gpto4\Finished"
OUTPUT_FILE         = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4_Confirmed_Gpto4\FinalOutput\Gpt260_ClaudeConfirmed.txt"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_text_files(folder: str):
    return sorted(
        fn for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn)) and fn.lower().endswith(".txt")
    )

# load prompts
system_prompt = read_text(SYSTEM_PROMPT_FILE)
user_prompt   = read_text(USER_PROMPT_FILE)
text_files    = get_text_files(TEXT_FOLDER)
print(f"Found {len(text_files)} text files.")

# Anthropic headers
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
}

start_time = time.time()

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for idx, filename in enumerate(text_files, 1):
        file_path = os.path.join(TEXT_FOLDER, filename)
        file_text = read_text(file_path)

        # combine user prompt + file contents
        user_message = f"{user_prompt}\n\n{file_text}"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens_to_sample": 4096,
            "temperature": 0.0,
        }

        print(f"Processing {idx}/{len(text_files)}: {filename}")
        resp = requests.post(
            "https://api.anthropic.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        data = resp.json()
        if resp.ok:
            reply = data["choices"][0]["message"]["content"].strip()
        else:
            reply = f"Error {resp.status_code}: {data}"

        # write out
        outfile.write(f"Entry {idx}\nFile: {filename}\n{reply}\n")
        outfile.write("=" * 50 + "\n")

        print(f"✓ {filename}")

        # Wait 1 second before the next API call
        time.sleep(2)

elapsed = time.time() - start_time
print(f"Done in {elapsed:.1f}s — results in {OUTPUT_FILE}")
