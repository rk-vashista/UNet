import os
import torch
from metest import single_image_inference  # Assuming the function is in single_image_inference.py

# Set paths
TEST_DATA_PATH = "./data/test"
MODEL_PATH = "./models/unet.pth"
RESULT_PATH = "./Result"

# Device configuration
device = "cpu"

# Create the result directory if it doesn't exist
os.makedirs(RESULT_PATH, exist_ok=True)

# Loop through all images in the test directory
for image_name in os.listdir(TEST_DATA_PATH):
    # Check if the file does not contain "predict" in the name and is an image file
    if "predict" not in image_name and image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(TEST_DATA_PATH, image_name)
        
        # Set the output path for the result
        output_path = os.path.join(RESULT_PATH, f"result_{image_name}")
        
        # Run inference and save the result
        single_image_inference(image_path, MODEL_PATH, device)
        
        # Move the result to the Result folder
        os.rename('./res/result.png', output_path)
        
        print(f"Processed and saved result for {image_name}")
