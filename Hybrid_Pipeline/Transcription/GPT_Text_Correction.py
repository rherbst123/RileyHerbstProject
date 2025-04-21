import os
import base64
import requests
import time

# ——— CONFIG ———
API_KEY            = ""  # ← your OpenAI API key
MODEL_NAME         = "gpt-4o-mini"
SYSTEM_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
#USER_PROMPT_FILE   = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\TextCorrection.txt"
PARENT_FOLDER      = r"c:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4"
# ————————

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def determine_type(path):
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"

# load prompts once
system_prompt = read_text_file(SYSTEM_PROMPT_FILE)
#user_prompt   = read_text_file(USER_PROMPT_FILE)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

sep = "\n" + ("="*50) + "\n"
start = time.time()

for sub in sorted(os.listdir(PARENT_FOLDER)):
    subpath = os.path.join(PARENT_FOLDER, sub)
    if not os.path.isdir(subpath):
        continue

    images_dir  = os.path.join(subpath, "images")
    rawtext_dir = os.path.join(subpath, "rawText")
    if not os.path.isdir(images_dir) or not os.path.isdir(rawtext_dir):
        print(f"Skipping {sub!r}: missing ‘images’ or ‘rawText’")
        continue

    # map basename → filename
    img_files = {
        os.path.splitext(fn)[0]: fn
        for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg","jpeg",".png"))
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

    out_path = os.path.join(f"{subpath}_corrected_transcripts.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        #out.write(f"=== Subfolder: {sub} ===\n")
        for idx, base in enumerate(common, start=1):
            img_fn = img_files[base]
            txt_fn = txt_files[base]

            raw_text = read_text_file(os.path.join(rawtext_dir, txt_fn))
            img_path = os.path.join(images_dir, img_fn)
            b64      = encode_image_to_base64(img_path)
            mime     = determine_type(img_path)

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": raw_text},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]}
                ],
                "max_tokens": 4098,
                "temperature": 0.0,
                "seed": 42
            }

            print(f"[{sub}] Entry {idx}: {base}")
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, json=payload
            )
            data = resp.json()
            corrected = (
                data.get("choices") or [{}]
            )[0].get("message", {}).get("content", 
               "No valid response returned from API."
            )

            #out.write(f"\nEntry {img_path}\n")
            out.write(f"Image: {base}\n")
            #out.write(f"Raw text ({txt_fn}):\n{raw_text}\n\n")
            out.write(f"\n{corrected}\n")
            out.write(sep)
            time.sleep(2)

    print(f"→ Wrote results for '{sub}' to {out_path!r}")

elapsed = time.time() - start
print(f"All done in {elapsed:.1f}s.")