# SPDX-License-Identifier: Apache-2.0
import base64
import boto3
import json
import sys

def read_prompt_from_file(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as file:
        return file.read().strip()

def format_response(response_text):
    # Split the response by newlines and remove empty lines
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    # Format each title with a number
    formatted_titles = []
    for i, line in enumerate(lines, 1):
        if line.startswith('-'):
            line = line[1:].strip()
        formatted_titles.append(f"{i}. {line}")
    return '\n'.join(formatted_titles)

def main(image_path, prompt_file):
    # Read prompt from file
    prompt = read_prompt_from_file(prompt_file)
    
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
    )

    MODEL_ID = "amazon.nova-pro-v1:0"
    # Open the image you'd like to use and encode it as a Base64 string.
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")

    system_list = [{
        "text": "Your task is to serve as an OCR"
    }]

    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpg",
                        "source": {"bytes": base64_string},
                    }
                },
                {
                    "text": prompt
                }
            ],
        }
    ]

    inf_params = {"max_new_tokens": 4096, "top_p": 0.1, "top_k": 20, "temperature": 0.1}

    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())

    # Get the response text and format it
    content_text = model_response["output"]["message"]["content"][0]["text"]
    formatted_response = format_response(content_text)

    # Save the formatted response to a file
    output_file = 'response_output.txt'
    with open(output_file, 'w') as f:
        f.write(formatted_response)

    print("\n[Response saved to response_output.txt]")
    print(formatted_response)

if __name__ == "__main__":
    path_to_file = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\Images\\0000_C0000578F.jpg"
    prompt_file = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\Python\\Inputs\\1.4StrippedPrompt.txt"
    
    if len(sys.argv) > 1:
        path_to_file = sys.argv[1]
    if len(sys.argv) > 2:
        prompt_file = sys.argv[2]
        
    main(path_to_file, prompt_file)
