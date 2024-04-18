import os
import cv2
from ultralytics import YOLO
import numpy as np


def color_test(directory, color, opacity, classes):
    # List all files in the directory
    files = os.listdir(directory)

    # Load YOLO model
    segment_model = YOLO('yolov8n-seg.pt')  # load an official model
    predict_model = YOLO('yolov8n.pt')  # load an official model
    
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
        else:
            # Perform segmentation using YOLO
            segment_results = segment_model.predict(image, classes=classes)
            
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
            
            # # Save the annotated image
            # annotated_image_path = os.path.join(directory, f'annotated_{image_file}')
            # cv2.imwrite(annotated_image_path, annotated_image)
            
            # Predict using the annotated image
            predict_results = predict_model.predict(annotated_image)
            
            # Define the desired window size (width, height)
            window_size = (800, 600)
            # Resize the image to fit the window size
            resized_image = cv2.resize(predict_results[0].plot(), window_size)
            # Display the prediction
            cv2.imshow("YOLOv8", resized_image)
            cv2.waitKey(0)  # Wait for any key press
            cv2.destroyAllWindows()  # Close OpenCV windows

if __name__ == '__main__':
    # color tuple format: (BLUE, GREEN, RED)
    # color_test('imgs/vase', (88, 219, 255), 0.1, classes=[75])
    # color_test('imgs/fork', (255, 0, 0), 0.1, classes=[42])
    # color_test('imgs/surfboard', (0, 0, 255), 0.1, classes=[37])
    # color_test('imgs/donut', (0, 255, 0), 0.1, classes=[54])
    # color_test('imgs/stopsign', (0, 0, 0), 0, classes=[11])
    # color_test('datasets/coco8/images/val', (0, 0, 255), 0.1, classes=None)
    color_test('datasets/coco128/images/train2017', (0, 0, 255), 0.1, classes=None)