from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Run inference on 'bus.jpg' with arguments
model.predict('bus.jpg', save=True)