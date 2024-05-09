import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def color_and_save_with_labels(image_directory, labels_directory, output_directory, color, opacity, classes=None):
    """
    Colorize and save images with labeled objects.

    Args:
        image_directory (str): Directory containing input images.
        labels_directory (str): Directory containing label files for the images.
        output_directory (str): Directory to save the annotated images.
        color (tuple): RGB color tuple (e.g., (0, 255, 255) for yellow).
        opacity (float): Opacity level for drawing objects (0.0 for fully transparent, 1.0 for opaque).
        classes (list): List of class indices to include (default is None, includes all classes).

    Returns:
        None
    """
    # List all files in the image directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Initialize error count
    error_count = 0
    # Initialize list to store error messages
    error_messages = []
    # Initialize dictionary to store object counts
    object_counts = {}
    # Initialize the progress bar with the total number of image files
    with tqdm(total=len(image_files), desc="Processing Images", position=0, leave=True) as pbar:
        # Iterate over each image file
        for image_file in image_files:
            # Construct the full path to the image file
            image_path = os.path.join(image_directory, image_file)
            # Load the image
            image = cv2.imread(image_path)
            # Check if the image was loaded successfully
            if image is None:
                error_count += 1
                error_messages.append(f"Error: Unable to load image '{image_file}'.")
                continue
            # Create a copy of the image for annotation
            annotated_image = image.copy()
            # Construct the full path to the label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_directory, label_file)
            # Check if the label file exists
            if not os.path.exists(label_path):
                error_count += 1
                error_messages.append(f"Error: Label file '{label_file}' does not exist for image '{image_file}'.")
                continue
            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()
            # Iterate over each line in the label file
            for line in lines:
                parts = line.strip().split(' ')
                class_index = int(parts[0])
                # Increment object count for this class_index
                object_counts[class_index] = object_counts.get(class_index, 0) + 1
                # Check if classes is None or the class_index is in the provided classes list
                if classes is None or class_index in classes:
                    points = list(map(float, parts[1:]))
                    # Reshape points into list of tuples [(x1, y1), (x2, y2), ...]
                    points = [(int(points[i] * image.shape[1]), int(points[i + 1] * image.shape[0])) for i in range(0, len(points), 2)]
                    if opacity > 0.0:
                        # Initialize blank mask image of same dimensions for drawing the shapes
                        shapes = np.zeros_like(annotated_image, np.uint8)
                        cv2.fillPoly(shapes, [np.array(points)], color)
                        # Generate output by blending image with shapes image, using the shapes
                        # images also as mask to limit the blending to those parts
                        out = annotated_image
                        mask = shapes.astype(bool)
                        out[mask] = cv2.addWeighted(annotated_image, opacity, shapes, 1 - opacity, 0)[mask]
                    else:
                        cv2.fillPoly(annotated_image, [np.array(points)], color)
            # Construct the full path to save the annotated image
            output_path = os.path.join(output_directory, image_file)
            # Save the annotated image
            cv2.imwrite(output_path, annotated_image)
            
            # Update the progress bar
            pbar.update(1)
            # Update the progress bar postfix with error count
            pbar.set_postfix(error_count=error_count)
            
    # Print error messages at the end
    if error_messages:
        print("Encountered the following errors:")
        for error_message in error_messages:
            print(error_message)
            
    # Generate and save histogram or bar chart of object counts
    if object_counts:
        plt.bar(object_counts.keys(), object_counts.values())
        plt.xlabel('Class Index')
        plt.ylabel('Count')
        plt.title('Object Counts')
        plt.savefig('object_counts.png')
        plt.close()


def color_and_save(directory, segment_model, color, opacity, classes=None):
    """
    Colorize and save images segmented by the provided model.

    Args:
        directory (str): Directory containing input images.
        segment_model (object): Segmentation model object.
        color (tuple): RGB color tuple (e.g., (0, 255, 255) for yellow).
        opacity (float): Opacity level for drawing objects (0.0 for fully transparent, 1.0 for opaque).
        classes (list): List of class indices to include (default is None, includes all classes).

    Returns:
        None
    """
    # List all files in the directory
    files = os.listdir(directory)
    # Filter only the image files (you can adjust the extensions as needed)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(directory, image_file)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Unable to load image '{image_file}'.")
            continue

        # Perform segmentation using the provided model
        segment_results = segment_model.predict(image, classes=classes)
        if not segment_results:
            print(f"No objects detected in '{image_file}'.")
            continue

        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Iterate over each segmentation result
        for result in segment_results:
            if result.masks is not None and result.masks.xy is not None:
                masks = result.masks.xy
                for mask in masks:
                    if len(mask) > 2:  # Ensure there are enough points to draw a polygon
                        mask_int = mask.astype(int)  # Convert points to integer format
                        if opacity > 0.0:
                            # Initialize blank mask image of same dimensions for drawing the shapes
                            shapes = np.zeros_like(annotated_image, np.uint8)
                            cv2.fillPoly(shapes, [mask_int], color)
                            # Generate output by blending image with shapes image, using the shapes
                            # images also as mask to limit the blending to those parts
                            out = annotated_image
                            mask = shapes.astype(bool)
                            out[mask] = cv2.addWeighted(annotated_image, opacity, shapes, 1 - opacity, 0)[mask]
                        else:
                            cv2.fillPoly(annotated_image, [mask_int], color)
        
        # Save the annotated image
        cv2.imwrite(image_path, annotated_image)

if __name__ == '__main__':
    # Define directories for images, labels, and output
    image_directory = 'datasets/segment/coco128-seg/images/train2017'
    labels_directory = 'datasets/segment/coco128-seg/labels/train2017'
    output_directory = 'datasets/detect/coco128_colored/images/train2017'
    color = (0, 255, 255)  # Adjust as needed
    opacity = 0.0  # Adjust as needed
    
    # Apply colorization and save the results
    color_and_save_with_labels(image_directory, labels_directory, output_directory, color, opacity)
