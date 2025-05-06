import os, base64, time, json, requests


API_KEY            = ""
MODEL_NAME         = "o4-mini"
SYSTEM_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
PARENT_FOLDER      = r"c:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted_4_19_25_FourthRun_TextCorrection_Gpto4"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def mime_type(path: str) -> str:
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"

system_prompt = read_text(SYSTEM_PROMPT_FILE)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

sep = "\n" + "=" * 50 + "\n"
t0  = time.time()

for sub in sorted(os.listdir(PARENT_FOLDER)):
    sub_dir = os.path.join(PARENT_FOLDER, sub)
    if not os.path.isdir(sub_dir):
        continue

    img_dir  = os.path.join(sub_dir, "images")
    txt_dir  = os.path.join(sub_dir, "rawText")
    if not (os.path.isdir(img_dir) and os.path.isdir(txt_dir)):
        print(f"Skipping {sub}: missing images/rawText")
        continue

    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    txts = {os.path.splitext(f)[0]: f for f in os.listdir(txt_dir)
            if f.lower().endswith(".txt")}

    shared = sorted(imgs.keys() & txts.keys())
    if not shared:
        print(f"No matching basename pairs in {sub}")
        continue

    out_file = os.path.join(f"{sub_dir}_corrected_transcripts.txt")
    with open(out_file, "w", encoding="utf-8") as out:
        for i, base in enumerate(shared, 1):
            img_path = os.path.join(img_dir, imgs[base])
            txt_path = os.path.join(txt_dir, txts[base])

            raw_text = read_text(txt_path)
            b64      = encode_b64(img_path)
            mime     = mime_type(img_path)

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": raw_text}
                    ]}
                ],
                "temperature": 0.0,
                "seed": 42
            }

            print(f"[{sub}] {i}/{len(shared)} → {base}")
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, data=json.dumps(payload)
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]

            out.write(f"Image: {base}\n{answer}\n{sep}")
            time.sleep(2)          # gentle rate‑limit

    print(f"Wrote → {out_file}")

print(f"Finished in {time.time() - t0:.1f} s")
