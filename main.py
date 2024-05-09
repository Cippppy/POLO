from ultralytics import YOLO
import wandb
import pandas as pd
import json
import yaml
import numpy as np
from color_and_save import color_and_save_with_labels
from cfg.colors import colors_dict


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')
    data_path = 'cfg/datasets/detect/val2017.yaml'
    project_name = "POLO11"
    test_img = "/home/cipoll17/POLO/datasets/detect/val2017/images/000000012639.jpg"
    classes = None
    device = '0'
    # Initialize a new wandb run for each color
    wandb.init(project=project_name, name=f"Baseline")
    
    # Customize validation settings for normal images
    validation_results = model.val(data=data_path,
                                   imgsz=640,
                                   batch=16,
                                   conf=0.25,
                                   iou=0.6,
                                   plots=True,
                                   save_json=True,
                                   device=device)
    wandb.log({"Example Image": wandb.Image(test_img)})
    ex_results = model.predict(test_img)[0].plot()
    ex_results = ex_results[:, :, ::-1]
    wandb.log({"Predicted Image": wandb.Image(ex_results)})
    map50_95 = validation_results.box.map    # map50-95
    map50 = validation_results.box.map50  # map50
    map75 = validation_results.box.map75  # map75
    map_by_class = validation_results.box.maps.tolist()
    wandb.log({"map50_95": map50_95, "map50": map50, "map75": map75})
    
    data_path = 'cfg/datasets/detect/colored/val2017-color.yaml'
    # Define directories for images, labels, and output
    image_directory = 'datasets/segment/val2017-seg/images'
    labels_directory = 'datasets/segment/val2017-seg/labels'
    output_directory = 'datasets/detect/val2017_colored/images'
    
    # End the current wandb run
    wandb.finish()
    # Define colors and opacities
    colors = list(colors_dict.values())
    opacities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_img = "/home/cipoll17/POLO/datasets/detect/val2017_colored/images/000000012639.jpg"
    # Loop over each color and opacity combination
    for color in colors:
        for opacity in opacities:
            # Skip if opacity > 0 and color is black (no coloring applied)
            if (opacity > 0) & (color == (0, 0, 0)):
                continue
            color_string = list(colors_dict.keys())[list(colors_dict.values()).index(color)]
            # Apply colorization and save the results
            color_and_save_with_labels(image_directory, labels_directory, output_directory, color, opacity, classes=classes)
            # Initialize a new wandb run for each color
            wandb.init(project=project_name, name=f"{color_string}_{int(opacity*100)}")
            
            # Customize validation settings for normal images
            validation_results = model.val(data=data_path,
                                        imgsz=640,
                                        batch=16,
                                        conf=0.25,
                                        iou=0.6,
                                        plots=True,
                                        save_json=True,
                                        device=device)
            wandb.log({"Example Image": wandb.Image(test_img)})
            ex_results = model.predict(test_img)[0].plot()
            ex_results = ex_results[:, :, ::-1]
            wandb.log({"Predicted Image": wandb.Image(ex_results)})
            map50_95 = validation_results.box.map    # map50-95
            map50 = validation_results.box.map50  # map50
            map75 = validation_results.box.map75  # map75
            map_by_class = validation_results.box.maps.tolist()
            wandb.log({"map50_95": map50_95, "map50": map50, "map75": map75})

            # End the current wandb run
            wandb.finish()
    