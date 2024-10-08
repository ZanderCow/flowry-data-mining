import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from contextlib import contextmanager
import uuid
import os
"""
Image Scraping Script with Selenium

This script scrapes images from a specified URL for a database website using Selenium and saves them in a directory named after a specified flower. 
The images are saved with filenames that include a unique UUID for easy identification.

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
"""

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
    Navigates to the specified URL, locates image elements, and saves each image to the directory named after the flower.
    
    Parameters:
        driver (selenium.webdriver.remote.webdriver.WebDriver): The WebDriver instance to use for navigating and interacting with the web page.
    
    Saves:
        Images located by the XPATH selector into the designated folder, with filenames following the format <flower_name>_<UUID>.png.
    
    Prints:
        The number of images found and the path where each image is saved.
    """
    driver.get(args.url)
    
    image_elements = driver.find_elements(By.XPATH, '//div[@class="col"]//img[@class="card-img-top"]')

    # Get the number of image elements
    num_images_to_download = len(image_elements)
    print(f"Found {num_images_to_download} images.")

    # Loop through each <img> element
    for index, image_element in enumerate(image_elements):
        # Generate a unique UUID for the image filename
        unique_filename = f'{args.flower_name}_{uuid.uuid4()}.png'
        image_path = os.path.join(save_directory, unique_filename)
        
        # Take a screenshot of the element
        image_element.screenshot(image_path)

        print(f"Saved image {index + 1} at {image_path}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Scrape images from a URL and save them with a specific flower name.')
parser.add_argument('url', type=str, help='The URL to scrape images from')
parser.add_argument('flower_name', type=str, help='The name of the flower (used for folder and file names)')
args = parser.parse_args()

url = "https://webcache.googleusercontent.com/search?q=cache:" + args.url
selenium_hub = "http://localhost:4444"

script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'flowers' directory
flowers_directory = os.path.join(script_directory, 'flowers')

# Create a directory with the flower name inside the 'flowers' directory
save_directory = os.path.join(flowers_directory, args.flower_name)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Usage of the context manager
if __name__ == "__main__":
    with get_driver() as driver:
        load_page_and_print_contents(driver)