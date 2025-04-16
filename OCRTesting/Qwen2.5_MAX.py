import os
import base64
from openai import OpenAI
import json

# Initialize the OpenAI client
client = OpenAI(
    api_key="sk-3787bb49f3934ffd82a3732b7ae7565b",  # Replace with your actual API key
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Function to encode an image file to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to format the response
def format_response(image_names, response_data):
    # Check if 'choices' is present and not empty in the response
    if "choices" in response_data and response_data["choices"]:
        content = response_data["choices"][0].get("message", {}).get("content", "")
        formatted_output = f"Images: {', '.join(image_names)}\n\n{content}\n"
    else:
        # If 'choices' is empty or not present, set a default message
        formatted_output = f"Images: {', '.join(image_names)}\n\nNo data returned from API.\n" 
    return formatted_output

# Function to read prompt file
def read_prompt_file(prompt_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Paths to your local image files
local_image_paths = [
    "C:\\Users\\riley\\Desktop\\10ImagesCollage\\0001_V0573776F_collage.png",  # Replace with your first image file path
    "C:\\Users\\riley\\Desktop\\10ImagesFull\\0001_V0573776F.jpg"   # Replace with your second image file path
]

# Path to the prompt file
prompt_file_path = "C:\\Users\\riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\Prompt_1.5.2.txt"  # update with your .txt file path

# Read the prompt text using the helper function
prompt_text = read_prompt_file(prompt_file_path).strip()

# Encode both local images to base64
base64_images = [encode_image_to_base64(path) for path in local_image_paths]

# Create the completion request with both images
completion = client.chat.completions.create(
    model="qwen-vl-max",  # You can change the model name as needed
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_images[0]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_images[1]}"
                    }
                }
            ]
        }
    ]
)

# Convert the response to a dictionary
response_data = json.loads(completion.model_dump_json())

# Extract the image file names
image_names = [os.path.basename(path) for path in local_image_paths]

# Format and print the response
formatted_output = format_response(image_names, response_data)
print(formatted_output)