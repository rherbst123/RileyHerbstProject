from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "C:\\Users\\Riley\\Desktop\\TestSet\\0000_C0000578F.jpg"

# Run inference on the source
results = model(source)