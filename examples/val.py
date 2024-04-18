from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')

    # data_path = 'cfg/datasets/detect/colored/coco8-color.yaml'
    # data_path = 'cfg/datasets/detect/colored/coco128-color.yaml'
    data_path = 'cfg/datasets/detect/coco128.yaml'
    # data_path = 'cfg/datasets/segment/coco8-seg.yaml'
    # data_path = 'cfg/datasets/segment/coco128-seg.yaml'
    
    # Customize validation settings
    validation_results = model.val(data=data_path,
                                   imgsz=640,
                                   batch=16,
                                   conf=0.25,
                                   iou=0.6,
                                   device='0')