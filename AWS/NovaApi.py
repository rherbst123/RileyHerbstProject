
import base64
import boto3
import json
import sys

#This is a testbed to see model performance and response for AWS Bedrock Models

def main(image_path, prompt,):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1", #Make sure us-east-1
    )

    MODEL_ID = "us.amazon.nova-pro-v1:0"
 
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")
    # Define your system prompt(s).
    system_list = [    {
            "text": "You are an expert artist. When the user provides you with an image, provide 3 potential art titles"
        }
    ]
    
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
    # Invoke the model and extract the response body.
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())
    # Pretty print the response JSON.
    print("[Full Response]")
    print(json.dumps(model_response, indent=2))
    # Print the text content for easy readability.
    content_text = model_response["output"]["message"]["content"][0]["text"]
    print("\n[Response Content Text]")
    print(content_text)

if __name__ == "__main__":
    path_to_file = "C:\\Users\\riley\\Desktop\\Field_museum_label_image.jpg"
    prompt = "Tell me what you see"
    if len(sys.argv) > 1:
        path_to_file = sys.argv[1]
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    if len(sys.argv) > 3:
        format = sys.argv[3]
    main(path_to_file, prompt)
