from ollama import chat
import os


prompt = "Prompts/Prompt_1.5_Ollama.txt"
# Read the prompt from input.txt
with open(prompt, 'r', encoding="utf-8") as file:
  prompt = file.read().strip()

folder_path = "c:\\Users\\Riley\\Desktop\\TestSet"
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

for image_path in image_files:
  response = chat(
    model='llama3.2-vision:11b',
    messages=[
      {
        'role': 'user',
        'content': prompt,
        'images': [image_path],
      }
    ],
  )
  with open("C:\\Users\\riley\\Documents\\GitHub\\RileyHerbstProject\\Outputs\\OllamaClient2_8_25_0403.txt", "a", encoding="utf-8") as output_file:
    output_file.write(f"Image: {image_path}\n")
    output_file.write(response.message.content + "\n")
    output_file.write("="*50 + "\n\n")