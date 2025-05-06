import os
import base64
import requests
import time


API_KEY            = ""
MODEL_NAME         = "claude-3-7-sonnet-20250219"
SYSTEM_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
#USER_PROMPT_FILE   = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\TextCorrection.txt"
PARENT_FOLDER      = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Base_Claude3.7"

#I hate working with claude almost as much as I hate eating whole logs of shit

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def determine_type(path: str) -> str:
    
    with open(path, "rb") as f:
        sig = f.read(8)
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if sig.startswith(b"\xFF\xD8"):          # JPEG
        return "image/jpeg"
    
    return "image/jpeg"                       

# load prompts once
system_prompt = read_text_file(SYSTEM_PROMPT_FILE)
#user_prompt   = read_text_file(USER_PROMPT_FILE)

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
}

sep   = "\n" + ("=" * 50) + "\n"
start = time.time()

for sub in sorted(os.listdir(PARENT_FOLDER)):
    subpath = os.path.join(PARENT_FOLDER, sub)
    if not os.path.isdir(subpath):
        continue

    images_dir  = os.path.join(subpath, "images")
    rawtext_dir = os.path.join(subpath, "rawText")
    if not os.path.isdir(images_dir) or not os.path.isdir(rawtext_dir):
        print(f"Skipping {sub!r}: missing 'images' or 'rawText'")
        continue

    img_files = {
        os.path.splitext(fn)[0]: fn
        for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    txt_files = {
        os.path.splitext(fn)[0]: fn
        for fn in os.listdir(rawtext_dir)
        if fn.lower().endswith(".txt")
    }

    common = sorted(set(img_files) & set(txt_files))
    if not common:
        print(f"No matching name pairs in {sub!r}")
        continue

    out_path = f"{subpath}_corrected_transcripts.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, base in enumerate(common, 1):
            img_path = os.path.join(images_dir,  img_files[base])
            txt_path = os.path.join(rawtext_dir, txt_files[base])

            raw_text = read_text_file(txt_path)
            mime     = determine_type(img_path)
            b64      = encode_image_to_base64(img_path)

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64", "media_type": mime, "data": b64}},
                            {"type": "text",
                             "text": f"{system_prompt}\n\nRaw text:\n{raw_text}"},
                        ],
                    }
                ],
                "max_tokens": 4098,
                "temperature": 0.0,
            }

            print(f"[{sub}] Entry {idx}: {base}")
            try:
                r = requests.post("https://api.anthropic.com/v1/messages",
                                  headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    corrected = (
                        data.get("content", [{}])[0].get("text")
                        or "No text content returned."
                    )
                else:
                    print(f"API error {r.status_code}: {r.text[:200]}…")
                    corrected = f"API ERROR {r.status_code}"
            except Exception as e:
                print(f"Request failed: {e}")
                corrected = f"ERROR: {e}"

            out.write(f"Image: {base}\n\n{corrected}\n{sep}")

    print(f"→ Wrote results for '{sub}' to '{out_path}'")

print(f"All done in {time.time() - start:.1f}s.")
