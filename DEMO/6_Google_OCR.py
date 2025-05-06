import os
import shutil
from google.cloud import vision
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\northern-union-424914-q7-1f2ba21ae567.json"

def detect_text(path):
    client = vision.ImageAnnotatorClient()

    try:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return ""
    except Exception as e:
        print(f"Error reading image file {path}: {e}")
        return ""

    image = vision.Image(content=content)

    try:
        response = client.text_detection(image=image)
    except Exception as e:
        print(f"Error calling Google Vision API for {path}: {e}")
        return ""

    if response.error.message:
        print(f'Google Vision API error for {path}: {response.error.message}')
        return ""

    texts = response.text_annotations

    if texts:
        return texts[0].description
    else:
        print(f"    No text detected in {path}")
        return ""

base_folder = r'C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\DEMO\\Segmented_Images_Cleaned'

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")

        images_folder = os.path.join(subfolder_path, 'images')
        rawtext_folder = os.path.join(subfolder_path, 'rawText')

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(rawtext_folder, exist_ok=True)

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)

            if os.path.isdir(file_path):
                continue

            if file.lower().endswith('.png'):
                print(f"  Processing image: {file_path}")

                destination_img_path = os.path.join(images_folder, file)
                try:
                    shutil.copy(file_path, destination_img_path)
                except Exception as e:
                    print(f"    Error copying {file_path} to {destination_img_path}: {e}")
                    continue

                extracted_text = detect_text(file_path)

                base_name = os.path.splitext(file)[0]
                txt_filename = f"{base_name}.txt"
                txt_path = os.path.join(rawtext_folder, txt_filename)

                try:
                    with open(txt_path, "w", encoding="utf-8") as text_file:
                        text_file.write(extracted_text)
                    print(f"    Text written to: {txt_path}")
                except Exception as e:
                    print(f"    Error writing text file {txt_path}: {e}")

print("Processing complete.")
