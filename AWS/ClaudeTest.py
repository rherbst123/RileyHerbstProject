import base64
import boto3
import json
import sys
import os

def analyze_image_with_nova(image_path, question):
    """
    Analyzes a JPEG image using Amazon Nova Pro via Amazon Bedrock
    """
    # Initialize the Bedrock runtime client
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-2"
    )

    # Verify file extension
    if not image_path.lower().endswith(('.jpg', '.jpeg')):
        print("Error: Only JPEG images (.jpg or .jpeg) are supported")
        return

    # Read and encode the image
    try:
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            # Convert binary data to base64
            base64_data = base64.b64encode(binary_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Could not find image file: {image_path}")
        return
    except Exception as e:
        print(f"Error reading image file: {str(e)}")
        return

    # Create the request body for Nova Pro with correct message structure
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "data": base64_data
                        }
                    }
                ]
            }
        ],
        "textGenerationConfig": {
            "maxTokenCount": 2048,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.8
        }
    }

    try:
        # Call the Bedrock API with the inference profile ID
        response = client.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(request_body)
        )

        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract and print the response
        print("\nNova Pro's Response:")
        print("-----------------")
        if 'messages' in response_body and len(response_body['messages']) > 0:
            print(response_body['messages'][0].get('content', 'No response generated'))
        else:
            print('No response generated')

    except client.exceptions.ClientError as error:
        print(f"Error calling Bedrock: {error.response['Error']['Message']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> \"<question>\"")
        print("Example: python script.py image.jpg \"What do you see in this image?\"")
        print("Note: Only JPEG images (.jpg or .jpeg) are supported")
        sys.exit(1)

    image_path = sys.argv[1]
    question = sys.argv[2]

    # Validate file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    # Call the analysis function
    analyze_image_with_nova(image_path, question)

if __name__ == "__main__":
    main()
