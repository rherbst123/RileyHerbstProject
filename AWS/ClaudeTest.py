import boto3
from botocore.exceptions import ClientError

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

IMAGE_NAME = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\Images\\0000_C0000578F.jpg"

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

with open(IMAGE_NAME, "rb") as f:
    image = f.read()

user_message = "Which countries consume more than 1000 TWh from hydropower? Think step by step and look at all regions. Output in JSON."

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
print(response_text)