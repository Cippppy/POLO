from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')

    data_path = 'cfg/datasets/detect/coco128.yaml'
    
    # Customize validation settings
    validation_results = model.val(data=data_path,
                                   imgsz=640,
                                   batch=16,
                                   conf=0.25,
                                   iou=0.6,
                                   device='0')