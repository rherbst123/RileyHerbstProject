from ollama_ocr import OCRProcessor

ocr = OCRProcessor(model_name='llama3.2-vision:11b')
# Process an image
result = ocr.process_image(
    image_path="c:\\Users\\Riley\\Desktop\\Segmented10ImagesTesting\\BigTest\\0011_K000263710\\mask_1.png",
    format_type="json"  # Options: markdown, text, json, structured, key_value
)
print(result)
