from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

# Run inference on 'bus.jpg' with arguments
model.predict('bus.jpg', save=True)