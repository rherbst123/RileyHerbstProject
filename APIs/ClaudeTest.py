import os
import base64
import requests
import json
from dotenv import load_dotenv

# Load API key from environment variable

ANTHROPIC_API_KEY = "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"

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_claude(image_path, prompt):
    """Send an image and prompt to Claude API and return the response"""
    # API endpoint
    url = "https://api.anthropic.com/v1/messages"
    
    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare headers
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Prepare the request body
    payload = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Make the API call
    response = requests.post(url, headers=headers, json=payload)
    
    # Return the response
    return response.json()

if __name__ == "__main__":
    # Example usage
    image_path = "c:\\Users\\Riley\\Desktop\\TextCorrection\\textTest_7\\0001_V0573776F\\images\\0001_V0573776F_segmentation_visualization.png"
    prompt = "What can you see in this image?"
    
    result = analyze_image_with_claude(image_path, prompt)
    
    # Print the response content
    if "content" in result and len(result["content"]) > 0:
        for content in result["content"]:
            if content["type"] == "text":
                print(content["text"])
    else:
        print("Error or unexpected response format:", result)