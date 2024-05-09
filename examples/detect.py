from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Run inference on 'bus.jpg' with arguments
model.predict('examples/bus.jpg', save=True, line_width=5, 
                                    imgsz=640,
                                   conf=0.25,
                                   iou=0.6)