import boto3
import os
import time
from botocore.exceptions import ClientError

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
IMAGE_FOLDER = "c:\\Users\\Riley\\Desktop\\TestSet"
TEXT_FILE_PATH = "Prompts/Prompt 1.5.txt"

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

with open(TEXT_FILE_PATH, "r", encoding="utf-8") as file:
    user_message = file.read().strip()

image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    with open(image_file, "rb") as f:
        print("Processing:", image_file)
        image = f.read()

    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": image}}},
                {"text": user_message},
            ],
        }
    ]

    response = bedrock_runtime.converse(
        modelId=MODEL_ID,
        messages=messages,
    )
    response_text = response["output"]["message"]["content"][0]["text"]
    print(f"Response for {image_file}: {response_text}")
    
    print("Sleeping")
    time.sleep(5)