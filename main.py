from ultralytics import YOLO
from color_and_save import color_and_save_with_labels
from cfg.colors import colors_dict


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')
    data_path = 'cfg/datasets/detect/coco128.yaml'
    
    # Customize validation settings for normal images
    validation_results = model.val(data=data_path,
                                   imgsz=640,
                                   batch=16,
                                   conf=0.25,
                                   iou=0.6,
                                   device='0')
    
    data_path = 'cfg/datasets/detect/colored/coco128-color.yaml'
    # Define directories for images, labels, and output
    image_directory = 'datasets/segment/coco128-seg/images/train2017'
    labels_directory = 'datasets/segment/coco128-seg/labels/train2017'
    output_directory = 'datasets/detect/coco128_colored/images/train2017'
    
    # Define colors and opacities
    colors = list(colors_dict.values())
    # opacities = [0, 0.25, 0.5, 0.75]
    opacities = [0.25]
    
    # Loop over each color and opacity combination
    for color in colors:
        for opacity in opacities:
            # Skip if opacity > 0 and color is black (no coloring applied)
            if (opacity > 0) & (color == (0, 0, 0)):
                continue
                
            # Apply colorization and save the results
            color_and_save_with_labels(image_directory, labels_directory, output_directory, color, opacity)
            
            # Run validation after colorization
            validation_results = model.val(data=data_path,
                                   imgsz=640,
                                   batch=16,
                                   conf=0.25,
                                   iou=0.6,
                                   device='0')
    