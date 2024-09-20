"""
Image Optimization and Resizing Script

This script optimizes and resizes images in a specified folder. The images are resized to 225x225 pixels, maintaining their aspect ratio, and saved with a new filename that includes a unique UUID. The script also removes the original images after optimization.

Usage:
    python script_name.py <flower_name>

Arguments:
    flower_name (str): The name of the flower (used for folder and file names).

Example:
    python script_name.py "rose"

Dependencies:
    - argparse
    - Pillow (PIL)
    - os
    - uuid
"""

import argparse
from PIL import Image
import os
import uuid

def image_optimizer(image_folder, flower_name):
    """
    Optimizes and resizes images in a specified folder. The images are resized to 225x225 pixels,
    maintaining their aspect ratio, and saved with a new filename that includes a unique UUID.

    Parameters:
        image_folder (str): The path to the folder containing the images to be optimized.
        flower_name (str): The name of the flower (used as a prefix for the new filenames).

    Resizes:
        Images to 225x225 pixels while maintaining aspect ratio.

    Crops:
        Images to 225x225 pixels after resizing.

    Converts:
        Images to RGB format for saving as JPEG.

    Saves:
        The optimized image with a new filename that includes the flower name and a UUID.

    Removes:
        The original image file after successful optimization.

    Prints:
        Success message for each optimized image or an error message if optimization fails.
    """
    # Loop through each file in the directory
    for filename in os.listdir(image_folder):
        # Construct the full file path
        image_path = os.path.join(image_folder, filename)

        try:
            # Open the image
            with Image.open(image_path) as img:
                # Get the current size of the image
                original_width, original_height = img.size

                # Calculate the new size maintaining the aspect ratio
                if original_width > original_height:
                    new_height = 225
                    new_width = int((new_height / original_height) * original_width)
                else:
                    new_width = 225
                    new_height = int((new_width / original_width) * original_height)

                # Resize the image while maintaining aspect ratio
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Calculate the coordinates to crop the image to 225x225
                left = (new_width - 225) / 2
                top = (new_height - 225) / 2
                right = (new_width + 225) / 2
                bottom = (new_height + 225) / 2

                # Perform the crop operation
                img = img.crop((left, top, right, bottom))

                # Convert image to RGB (necessary for JPEG)
                img = img.convert("RGB")

                # Generate a unique filename
                unique_filename = f'{flower_name}_{uuid.uuid4()}.jpg'
                new_image_path = os.path.join(image_folder, unique_filename)

                # Compress and save the image with the new filename
                img.save(new_image_path, "JPEG", optimize=True, quality=85)

                # Remove the original file
                os.remove(image_path)

                print(f"Optimized, resized, and saved {filename} to {unique_filename} successfully.")
        
        except Exception as e:
            print(f"Failed to optimize {filename}: {e}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Optimize and resize images in a specific folder with a flower name prefix.')
    parser.add_argument('flower_name', type=str, help='The name of the flower (used for folder and file names)')
    args = parser.parse_args()

    # Set up directories
    script_directory = os.path.dirname(os.path.abspath(__file__))
    flowers_directory = os.path.join(script_directory, 'flowers')
    image_folder = os.path.join(flowers_directory, args.flower_name)

    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Error: The folder {image_folder} does not exist.")
        sys.exit(1)

    # Optimize the images
    image_optimizer(image_folder, args.flower_name)