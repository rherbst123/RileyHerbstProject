import os
import base64
import requests
import time

# --- Configuration variables ---
API_KEY = ""
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\SystemPrompt.txt"
USER_PROMPT_FILE = r"C:\Users\Riley\Documents\GitHub\RileyHerbstProject\Prompts\Prompt_1.5.2.txt"
COMBINED_PROMPT_FILE = r"C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\Prompts\\CombinedPrompt.txt"


# Set the base folder which contains subfolders (each with images)
BASE_IMAGE_FOLDER = r"C:\\Users\\Riley\\Desktop\\300Images_Segmented_Conversation_History\\Segmented"

# Separator between entries in the text file.
separator = "=" * 50 + "\n"

# --- Helper functions ---
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def determine_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    else:
        return "image/jpeg"

def get_image_files(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png")
    files = [filename for filename in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(valid_extensions)]
    return sorted(files)


# Updated run_combined_prompt function
def run_combined_prompt(conversation_text):
    # Read the prompt template from file
    base_prompt = read_text_file(COMBINED_PROMPT_FILE)
    
    # Inject the conversation into the prompt
    combined_input = base_prompt.replace("{conversation_text}", conversation_text)
    
    messages = [
        {
            "role": "system",
            "content": "Your job is to combine a set of outputs into one combined list."
        },
        {
            "role": "user",
            "content": combined_input
        }
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4098,
        "temperature": 0.0,
        "seed": 42
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()

    if "choices" in response_data and response_data["choices"]:
        combined_reply = response_data["choices"][0].get("message", {}).get("content", "")
    else:
        combined_reply = "No valid combined prompt generated."
    return combined_reply

# --- Read the prompt files once ---
system_prompt = read_text_file(SYSTEM_PROMPT_FILE)
user_prompt_base = read_text_file(USER_PROMPT_FILE)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

start_time = time.time()

# Initialize an accumulator for all folder combined prompts.
final_combined_entries = ""

# --- Process each subfolder ---
subfolders = [d for d in os.listdir(BASE_IMAGE_FOLDER)
              if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, d))]

print(f"Found {len(subfolders)} subfolders in the base folder.")

for subfolder in subfolders:
    folder_path = os.path.join(BASE_IMAGE_FOLDER, subfolder)
    image_files = get_image_files(folder_path)
    print(f"\nProcessing folder '{subfolder}' with {len(image_files)} images.")
    
    # Create a conversation history file in the subfolder.
    conversation_history_file = os.path.join(folder_path, "conversation_history.txt")
    
    # Initialize conversation history (will accumulate previous responses).
    conversation_history = ""
    
    with open(conversation_history_file, "w", encoding="utf-8") as outfile:
        for idx, img_filename in enumerate(image_files, start=1):
            image_path = os.path.join(folder_path, img_filename)
            encoded_image = encode_image_to_base64(image_path)
            image_type = determine_type(image_path)
            
            # Construct the dynamic user prompt by appending previous conversation history.
            dynamic_user_prompt = user_prompt_base + "\n" + conversation_history
            
            # Build the message content with text and image.
            message_content = [
                {
                    "type": "text",
                    "text": dynamic_user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_type};base64,{encoded_image}"
                    }
                }
            ]
            
            # Build the complete message list.
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 4098,
                "temperature": 0.0,
                "seed": 42
            }
            
            print(f"Processing image {idx} in folder '{subfolder}': '{img_filename}'")
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_data = response.json()
            
            if "choices" in response_data and response_data["choices"]:
                assistant_reply = response_data["choices"][0].get("message", {}).get("content", "")
            else:
                assistant_reply = "No valid response returned from API."
            
            # Append the new reply to the conversation history.
            conversation_history += f"\nEntry {idx} (Image: {img_filename}):\n{assistant_reply}\n"
            
            # Write the current entry to the conversation history file.
            entry_text = (
                f"Entry {idx}\n"
                f"Image: {img_filename}\n"
                f"{assistant_reply}\n"
                f"{separator}"
            )
            outfile.write(entry_text)
            outfile.flush()
            print(f"Image {idx} processed in folder '{subfolder}'.")
    
    print(f"Conversation history for folder '{subfolder}' saved to {conversation_history_file}")
    
    # Run the combined prompt function using the accumulated conversation history.
    combined_prompt = run_combined_prompt(conversation_history)
    combined_output_file = os.path.join(folder_path, "combined_prompt.txt")
    with open(combined_output_file, "w", encoding="utf-8") as combined_file:
        combined_file.write(combined_prompt)
    print(f"Combined prompt for folder '{subfolder}' saved to {combined_output_file}")
    
    # Add the combined prompt to the final accumulator with a header separator for clarity.
    final_combined_entries += f"Image: {subfolder}\n{'-'*30}\n{combined_prompt}\n{'='*50}\n"

# After processing all folders, write the final combined entries to a single file.
final_output_file = os.path.join(BASE_IMAGE_FOLDER, "final_combined_prompt.txt")
with open(final_output_file, "w", encoding="utf-8") as final_file:
    final_file.write(final_combined_entries)
print(f"\nFinal combined prompt for all folders saved to {final_output_file}")

elapsed_time = time.time() - start_time
minutes = elapsed_time // 60
print(f"\nAll folders processed in {minutes:.2f} minutes.")
