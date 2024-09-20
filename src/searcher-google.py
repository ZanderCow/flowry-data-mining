"""
Image Scraping, Cropping, and Scaling Script with Selenium and PIL

This script scrapes images from a specified URL using Selenium, scrolls through the page to load more images, and then downloads, resizes, and saves the images in a directory named after a specified flower. The images are resized to 225x225 pixels and saved with a unique filename.

Usage:
    python script_name.py <url> <flower_name>

Arguments:
    url (str): The URL to scrape images from.
    flower_name (str): The name of the flower (used for folder and file names).

Example:
    python searcher-google.py "https://example.com/page" "rose"

Dependencies:
    - argparse
    - selenium
    - requests
    - Pillow (PIL)
    - contextlib
    - uuid
    - os
    - time

Scrolling:
    The script scrolls the page for a specified amount of time to load more images dynamically.

Image Processing:
    The script downloads each image, resizes it to 225x225 pixels using the LANCZOS resampling filter, and saves it in the designated folder with a filename that includes the flower name and a unique UUID.

Notes:
    - The Selenium WebDriver is configured to run on a remote Selenium server (`selenium_hub`).
    - Ensure that the remote Selenium server is running and accessible at the specified URL.
    - The images are processed using the PIL (Pillow) library, which handles resizing and saving.

Error Handling:
    The script includes error handling for image downloading and processing. If an image fails to download or process, an error message is printed, and the script continues to the next image.
"""

import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from contextlib import contextmanager
import uuid
import os
import time
import requests
from PIL import Image
from io import BytesIO

# Parse command line arguments
parser = argparse.ArgumentParser(description='Scrape images from a URL, crop/scale them, and save them with a specific flower name.')
parser.add_argument('url', type=str, help='The URL to scrape images from')
parser.add_argument('flower_name', type=str, help='The name of the flower (used for folder and file names)')
args = parser.parse_args()

url = args.url
selenium_hub = "http://localhost:4444"

script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'flowers' directory
flowers_directory = os.path.join(script_directory, 'flowers')
flower_name = args.flower_name

# Create a directory with the flower name inside the 'flowers' directory
save_directory = os.path.join(flowers_directory, flower_name)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Set the amount of time to scroll in seconds
scroll_time = 15  # Set this to the desired scrolling time

@contextmanager
def get_driver():
    """
    A context manager that initializes and returns a Selenium WebDriver instance configured to run on a remote Selenium server.
    
    Yields:
        selenium.webdriver.remote.webdriver.WebDriver: A Selenium WebDriver instance.
    
    Raises:
        None. Ensures that the WebDriver instance is quit after usage.
    """
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Remote(
        command_executor=selenium_hub,
        options=chrome_options
    )
    try:
        yield driver
    finally:
        driver.quit()

def load_page_and_save_images(driver):
    """
    Navigates to the specified URL, scrolls down the page to load images, and processes each found image.

    Parameters:
        driver (selenium.webdriver.remote.webdriver.WebDriver): The WebDriver instance to use for navigating and interacting with the web page.
    
    Scrolls:
        The page for a specified amount of time to dynamically load more content.

    Downloads:
        Each image from its `src` attribute.

    Processes:
        Crops each image to a square (shortest side x shortest side), then resizes it to 255x255 pixels.

    Saves:
        The processed images in the designated folder with filenames following the format <flower_name>_<UUID>.jpg.

    Prints:
        A success message for each processed image or an error message if the image fails to download or process.
    """
    driver.get(url)
    
    scroll_pause_time = 0.5  # Pause time between scrolls
    end_time = time.time() + scroll_time  # End time based on the scroll_time variable
    while time.time() < end_time:
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(scroll_pause_time)

    # Find image elements
    img_elements = driver.find_elements(By.CSS_SELECTOR, 'div.H8Rx8c img')

    print(f"Found {len(img_elements)} images")

    for i, element in enumerate(img_elements):
        img_src = element.get_attribute('src')

        try:
            # Download the image
            response = requests.get(img_src)
            response.raise_for_status()  # Raise an error for bad responses

            # Open the image using PIL
            img = Image.open(BytesIO(response.content))

            # Determine the shortest side
            min_side = min(img.size)

            # Calculate the coordinates for cropping the image to a square
            left = (img.width - min_side) / 2
            top = (img.height - min_side) / 2
            right = (img.width + min_side) / 2
            bottom = (img.height + min_side) / 2

            # Crop the image to a square
            img = img.crop((left, top, right, bottom))

            # Resize the cropped image to 255x255 pixels
            img = img.resize((255, 255), Image.Resampling.LANCZOS)

            # Save the final image
            unique_filename = f'{flower_name}_{uuid.uuid4()}.jpg'
            image_path = os.path.join(save_directory, unique_filename)
            img.save(image_path)

            print(f"Saved and resized image {i+1} as {unique_filename}")

        except Exception as e:
            print(f"Failed to process image {i+1} from {img_src}: {e}")
# Usage of the context manager
if __name__ == "__main__":
    with get_driver() as driver:
        load_page_and_save_images(driver)