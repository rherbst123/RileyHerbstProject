from ollama_ocr import OCRProcessor

#ocr = OCRProcessor(model_name='llama3.2-vision:11b')
ocr = OCRProcessor(model_name='granite3.2-vision')
# Process an image
result = ocr.process_image(
    image_path="c:\\Users\\Riley\Desktop\\300ImagesSegmentted4_14_25_ThirdRun_OCRd\\0074_V0228073F\\mask_6.png",
    format_type="text"  # Options: markdown, text, json, structured, key_value
)
print(result)
