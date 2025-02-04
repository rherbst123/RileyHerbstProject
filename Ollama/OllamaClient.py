from ollama import chat

path = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\Images\\0000_C0000578F.jpg"
#need to have ollama running in background with model downloaded and loaded :)
response = chat(
  model='llama3.2-vision',
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image? Be concise.',
      'images': [path],
    }
  ],
)

print(response.message.content)