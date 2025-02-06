from ollama import chat
import os


prompt = "Prompts/Prompt 1.5.txt"
# Read the prompt from input.txt
with open(prompt, 'r', encoding="utf-8") as file:
  prompt = file.read().strip()

folder_path = "c:\\Users\\Riley\\Desktop\\TestSet"
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

for image_path in image_files:
  response = chat(
    model='llama3.2-vision',
    messages=[
      {
        'role': 'user',
        'content': prompt,
        'images': [image_path],
      }
    ],
  )
  print(f"Image: {image_path}")
  print(response.message.content)
  print("="*50)
  print()