import os
import base64
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="",  # Replace with your actual API key
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Function to encode an image file to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to read a single prompt from a .txt file with explicit UTF-8 encoding
def read_single_prompt_from_file(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as file:  # Explicitly specify UTF-8 encoding
        prompt = file.read().strip()  # Read the entire file content as a single prompt
    return prompt

# Function to process a folder of images and write responses to a text file
def process_images_in_folder(folder_path, output_file, prompt_file=None):
    # Read the single prompt from the .txt file if provided
    prompt = read_single_prompt_from_file(prompt_file) if prompt_file else "What is this?"

    # Open the output file in append mode
    with open(output_file, "a", encoding="utf-8") as outfile:  # Explicitly specify UTF-8 encoding
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)
            
            # Skip if it's not a file or not an image
            if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {filename}")
                continue
            
            print(f"Processing image: {filename}")
            
            try:
                # Encode the local image to base64
                base64_image = encode_image_to_base64(image_path)

                # Create the completion request
                completion = client.chat.completions.create(
                    model="qwen-vl-max",  # You can change the model name as needed
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"  # Use base64 encoding for the local image
                                    }
                                }
                            ]
                        }
                    ]
                )

                # Extract the response from the completion
                response_text = completion.choices[0].message.content

                # Write the result to the output file
                outfile.write(f"Image: {filename}\n")
                outfile.write(f"-"*30)
                outfile.write(f"\n")
                #outfile.write(f"Prompt: {prompt}\n")
                outfile.write(f"{response_text}\n")
                outfile.write("=" * 50 + "\n")

                print(f"{filename},\n{response_text}")

            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                outfile.write(f"Error processing image: {filename}. Error: {str(e)}\n")
                outfile.write("=" * 50 + "\n")

# Path to your folder containing images
folder_path = "C:\\Users\\riley\\Desktop\\10ImagesFull"  # Replace with the actual path to your folder

# Path to your .txt file containing prompts (optional)
prompt_file = "C:\\Users\\riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\Prompt_1.5.2.txt"  # Replace with the actual path to your prompt file or set to None

# Output file to save responses
output_file = "C:\\Users\\riley\\Desktop\\Qwen2.5_Max_4_15_25_Test_3.txt"  # Name of the output text file

# Process the images in the folder
process_images_in_folder(folder_path, output_file, prompt_file)

print(f"All images processed. Responses saved to {output_file}")