import os
import cv2
from ultralytics import YOLO

# Directory containing the images
directory = 'imgs/vase'
# directory = 'imgs/vases'

# Load YOLO model
segment_model = YOLO('yolov8n-seg.pt')  # load an official model
predict_model = YOLO('yolov8n.pt')  # load an official model

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
    else:
        # Perform segmentation using YOLO
        segment_results = segment_model.predict(image, classes=[75])
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Iterate over each segmentation result
        for result in segment_results:
            if result.masks is not None and result.masks.xy is not None:
                masks = result.masks.xy
                for mask in masks:
                    if len(mask) > 2:  # Ensure there are enough points to draw a polygon
                        mask_int = mask.astype(int)  # Convert points to integer format
                        cv2.fillPoly(annotated_image, [mask_int], (255, 255, 0))
        
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
