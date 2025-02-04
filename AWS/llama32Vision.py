import boto3
import json
import base64
from botocore.config import Config
import os

# Initialize the Bedrock client
config = Config(
    region_name=os.getenv("BEDROCK_REGION", "us-west-2"),
)
bedrock_runtime = boto3.client('bedrock-runtime', config=config)
MODEL_ID = "us.meta.llama3-2-90b-instruct-v1:0"

# Read and encode the image
image_path = "C:\\Users\\riley\\Desktop\\Field_museum_label_image.jpg"  # Replace with the actual path to your image
try:
    # Open the image file and read its contents
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    # Encode the image bytes to base64
    image_data = image_bytes
except FileNotFoundError:
    print(f"Image file not found at {image_path}")
    image_data = None

# Construct the messages for the model input
messages = [
    {
        "role": "user",
        "content": [
            {
                "text": "Tell me what you see here"
            },
            {
                "image": {
                    "format": ".jpg",
                    "source": {
                        "bytes": image_data
                    }
                }
            }
        ]
    }
]

try:
    # Invoke the SageMaker endpoint
    response = bedrock_runtime.converse(
        modelId=MODEL_ID,  # MODEL_ID defined at the beginning
        messages=[messages],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0,
            "topP": .1
        },
    )

    # Read the response
    print(response['output']['message']['content'][0]['text'])

except Exception as e:
    print(f"An error occurred while invoking the endpoint: {str(e)}")
