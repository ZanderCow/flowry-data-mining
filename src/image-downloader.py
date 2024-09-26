import requests
from PIL import Image
from io import BytesIO
import os
import json
import concurrent.futures

# Specify the output folder (update this path to your desired output directory)
output_folder = '../flowers/tulip'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Path to your JSON file containing the list of image URLs
json_file_path = '../flowers/tulip/Tulip_img_src.json'

# Desired crop size
crop_size = 256

# Desired image quality for compression (1-95)
image_quality = 85  # Adjust this value as needed (lower means more compression)

# Read the list of image URLs from the JSON file
with open(json_file_path, 'r') as f:
    url_list = json.load(f)

def process_image(idx_url):
    idx, url = idx_url
    try:
        # Download the image
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Ensure the request was successful

        # Open the image from the response content
        img = Image.open(BytesIO(response.content))

        # Convert image to RGB if it's in a different mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get original dimensions
        width, height = img.size

        # Calculate scaling factor to resize the image so that one side equals crop_size
        scale_factor = crop_size / min(width, height)

        # New dimensions after resizing
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image while maintaining aspect ratio
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Calculate coordinates for the central crop
        left = (new_width - crop_size) / 2
        top = (new_height - crop_size) / 2
        right = (new_width + crop_size) / 2
        bottom = (new_height + crop_size) / 2

        # Crop the center of the image to crop_size x crop_size
        img = img.crop((left, top, right, bottom))

        # Extract the image filename from the URL
        filename = os.path.basename(url) + '.jpg'

        # Full path for saving the image
        full_path = os.path.join(output_folder, filename)

        # Save the image in JPEG format with compression
        img.save(full_path, 'JPEG', quality=image_quality, optimize=True)

        print(f"Saved compressed image {full_path}")

    except Exception as e:
        print(f"Failed to process {url}: {e}")

# Use ThreadPoolExecutor to process images in parallel
max_workers = 20  # Adjust this number according to your system capabilities
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_image, enumerate(url_list))