import os
from google.cloud import vision
import io
import os


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:\\Users\\Riley\\Desktop\\Portal\\Code\\northern-union-424914-q7-1f2ba21ae567.json"

#Take notes Open Ai...
def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    print('Texts:')
    for text in texts:
        print(f'\n"{text.description}"')
       
        vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        print('bounds: {}'.format(','.join(vertices)))

    if texts:
        return texts[0].description
    else:
        return ""

if __name__ == '__main__':
    
    image_path = 'c:\\Users\\Riley\\Desktop\\10ImagesFull\\0001_V0573776F.jpg'
    
    
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
    else:
        full_text = detect_text(image_path)
        print("\nFull text from image:")
        print(full_text)
