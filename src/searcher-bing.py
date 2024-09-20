"""
Image Scraping Script For Bing Search 

This script scrapes images from a specified URL thats a bing image search one using Selenium, 
scrolls through the page for a set amount of time, 
and optionally saves images that meet certain criteria (e.g., size). 
The images are saved in a directory named after a specified flower.

Usage:
    python script_name.py <url> <flower_name>

Arguments:
    url (str): The URL to scrape images from.
    flower_name (str): The name of the flower (used for folder and file names).

Example:
    python script_name.py "example.com/page" "rose"

Dependencies:
    - argparse
    - selenium
    - contextlib
    - uuid
    - os
    - time

Scrolling:
    The script scrolls the page for a specified amount of time to load more images dynamically.

Image Filtering:
    Uncomment the relevant code sections to filter images based on their dimensions before saving.

Saving Images:
    The code currently prints the `src` attribute of the images instead of saving them. To save images, modify the code as needed.

Notes:
    - The Selenium WebDriver is configured to run on a remote Selenium server (`selenium_hub`).
    - Ensure that the remote Selenium server is running and accessible at the specified URL.
"""

import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from contextlib import contextmanager
import uuid
import os
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Scrape images from a URL and save them with a specific flower name.')
parser.add_argument('url', type=str, help='The URL to scrape images from')
parser.add_argument('flower_name', type=str, help='The name of the flower (used for folder and file names)')
args = parser.parse_args()

url = args.url
selenium_hub = "http://localhost:4444"

script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'flowers' directory
flowers_directory = os.path.join(script_directory, 'flowers')

# Create a directory with the flower name inside the 'flowers' directory
save_directory = os.path.join(flowers_directory, args.flower_name)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Set the amount of time to scroll in seconds
scroll_time = 3  # Set this to the desired scrolling time

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

def load_page_and_print_contents(driver):
    """
    Navigates to the specified URL, scrolls down the page to load images, and identifies image elements based on an XPath.

    Parameters:
        driver (selenium.webdriver.remote.webdriver.WebDriver): The WebDriver instance to use for navigating and interacting with the web page.
    
    Scrolls:
        The page for a specified amount of time to dynamically load more content.
    
    Finds:
        Images on the page by using the specified XPath for elements containing the `mimg` class and `th.bing.com` in the `src` attribute.
    
    Prints:
        The `src` attribute of each identified image.
    
    (Optional) Saves:
        Uncomment the relevant code sections to save images with filenames following the format <flower_name>_<UUID>.png if they meet certain criteria (e.g., size).
    """
    driver.get(args.url)

    # Scroll down for the specified amount of time
    scroll_pause_time = 0.5  # Pause time between scrolls
    end_time = time.time() + scroll_time  # End time based on the scroll_time variable
    while time.time() < end_time:
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(scroll_pause_time)

    # Wait for images to load (additional wait after scrolling)
    time.sleep(1)
    image_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'mimg') and contains(@src, 'th.bing.com')]")

    # Get the number of image elements
    num_images_to_download = len(image_elements)
    print(f"Found {num_images_to_download} images.")

    # Loop through each <img> element and print the src
    for index, image_element in enumerate(image_elements):
        # Get the src link of the image
        src = image_element.get_attribute("src")
        print(src)

        # (Optional) Uncomment the code below to save images based on their dimensions
        # width = image_element.size['width']
        # height = image_element.size['height']
        # if width > 250 and height > 250:
        #     unique_filename = f'{args.flower_name}_{uuid.uuid4()}.png'
        #     image_path = os.path.join(save_directory, unique_filename)
        #     image_element.screenshot(image_path)
        #     print(f"Saved image {index + 1} at {image_path}")
        # else:
        #     print(f"Skipped image {index + 1} with dimensions {width}x{height}")

# Usage of the context manager
if __name__ == "__main__":
    with get_driver() as driver:
        load_page_and_print_contents(driver)