# collect_img_src.py

import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    TimeoutException,
    NoSuchElementException
)
from contextlib import contextmanager
import os
import time

# -------------------------------
# Configuration Parameters
# -------------------------------

URL = "https://identify.plantnet.org/k-world-flora/species/Gypsophila%20paniculata%20L./data#galleries"
SELENIUM_HUB = "http://localhost:4444"  # Update if different
FLOWER_NAME = "baby's-breath"
TARGET_ALT = "Gypsophila paniculata Flower"
CLICKS = 50  # Number of times to click the "Load more" button

# -------------------------------
# Save Path Configuration
# -------------------------------

SAVE_PATH = "../flowers/babys-breath"  # <-- Update this path

# Initialize variables
json_path = None

if os.path.isdir(SAVE_PATH):
    # SAVE_PATH is a directory
    SAVE_DIRECTORY = SAVE_PATH
    if not os.path.exists(SAVE_DIRECTORY):
        try:
            os.makedirs(SAVE_DIRECTORY)
            print(f"Created save directory at '{SAVE_DIRECTORY}'.")
        except Exception as e:
            print(f"Failed to create save directory '{SAVE_DIRECTORY}': {e}")
            exit(1)
    else:
        print(f"Using existing directory '{SAVE_DIRECTORY}' for saving files.")
    # Construct the json filename
    json_filename = f"{FLOWER_NAME}_img_src.json"
    json_path = os.path.join(SAVE_DIRECTORY, json_filename)
elif os.path.isfile(SAVE_PATH) or SAVE_PATH.endswith('.json'):
    # SAVE_PATH is a file or ends with '.json', treat it as a JSON file
    json_path = SAVE_PATH
    print(f"Using JSON file '{json_path}' for saving URLs.")
else:
    # SAVE_PATH does not exist yet
    if SAVE_PATH.endswith('.json'):
        # It's a JSON file that doesn't exist yet, we can proceed
        json_path = SAVE_PATH
        print(f"Will create new JSON file '{json_path}' for saving URLs.")
    else:
        # Assume it's a directory that doesn't exist yet, create it
        SAVE_DIRECTORY = SAVE_PATH
        try:
            os.makedirs(SAVE_DIRECTORY)
            print(f"Created save directory at '{SAVE_DIRECTORY}'.")
        except Exception as e:
            print(f"Failed to create save directory '{SAVE_DIRECTORY}': {e}")
            exit(1)
        # Construct the json filename
        json_filename = f"{FLOWER_NAME}_img_src.json"
        json_path = os.path.join(SAVE_DIRECTORY, json_filename)

# -------------------------------
# Selenium WebDriver Setup
# -------------------------------

@contextmanager
def get_driver():
    """
    Context manager to initialize and quit the Selenium WebDriver.
    """
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--headless')

    driver = webdriver.Remote(
        command_executor=SELENIUM_HUB,
        options=chrome_options
    )
    try:
        yield driver
    finally:
        driver.quit()

# -------------------------------
# Image Collection Function
# -------------------------------

def load_page_and_collect_img_src(driver):
    """
    Load the target page, click the "Load more" button multiple times,
    locate all relevant <img> elements, extract their src attributes,
    and return the set of img src URLs.

    Args:
        driver (WebDriver): The Selenium WebDriver instance.

    Returns:
        set: A set of image URLs.
    """
    driver.get(URL)
    image_urls = set()  # Initialize the set to store image URLs

    try:
        # Click the "Load more" button multiple times to load all images
        for i in range(CLICKS):
            try:
                # Wait until the "Load more" button is clickable
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//div[@class='mt-3']//button[text()='Load more']")
                    )
                )
                # Scroll the button into view and click it using JavaScript
                driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                driver.execute_script("arguments[0].click();", load_more_button)
                print(f"Clicked 'Load more' button {i+1}/{CLICKS}")
                # Wait for new content to load; using explicit wait is better but depends on the page structure
                time.sleep(0.5)  # Consider replacing with a more reliable wait

            except (ElementClickInterceptedException, TimeoutException, NoSuchElementException) as e:
                print(f"'Load more' button not found or click intercepted on attempt {i+1}. Error: {e}. Exiting the loop.")
                break  # Exit the loop if unable to click or find the button

        # After loading all images, locate the desired <img> elements
        imgs = driver.find_elements(
            By.XPATH,
            f'//img[@alt="{TARGET_ALT}"]'
        )
        print(f"Found {len(imgs)} images with alt='{TARGET_ALT}'.")

        # Extract the src attribute from each <img>
        for index, img in enumerate(imgs, start=1):
            try:
                img_src = img.get_attribute('src')
                if img_src:
                    # Replace '/s/' with '/o/' in the URL
                    if '/s/' in img_src:
                        img_src = img_src.replace('/s/', '/o/')
                    if img_src not in image_urls:
                        image_urls.add(img_src)
                        print(f"Collected image {index}: {img_src}")
                    else:
                        print(f"Image {index} is a duplicate.")
                else:
                    print(f"Image {index} has no src attribute.")
            except Exception as e:
                print(f"Error extracting src for image {index}: {e}")

    except Exception as e:
        print(f"An error occurred while loading the page and collecting image links: {e}")

    return image_urls

# -------------------------------
# Load Existing URLs Function
# -------------------------------

def load_existing_urls(json_path):
    """
    Load existing URLs from the JSON file if it exists.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        set: A set of existing image URLs.
    """
    existing_urls = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, list):
                    existing_urls = set(data)
                    print(f"Loaded {len(existing_urls)} existing URLs from '{json_path}'.")
                else:
                    print(f"JSON file '{json_path}' does not contain a list. Proceeding with empty set.")
        except Exception as e:
            print(f"Failed to read existing URLs from '{json_path}': {e}")
    else:
        print(f"No existing JSON file at '{json_path}'. Starting with empty URL set.")
    return existing_urls

# -------------------------------
# Save to JSON Function
# -------------------------------

def save_img_src_to_json(image_urls, json_path):
    """
    Save the list of image URLs to a JSON file.

    Args:
        image_urls (list): List of image URLs.
        json_path (str): Path to the JSON file.
    """
    try:
        with open(json_path, 'w') as json_file:
            json.dump(image_urls, json_file, indent=4)
        print(f"Successfully saved {len(image_urls)} image URLs to '{json_path}'.")
    except Exception as e:
        print(f"Failed to save image URLs to JSON: {e}")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Load existing URLs if any
    existing_urls = load_existing_urls(json_path)

    with get_driver() as driver:
        image_links = load_page_and_collect_img_src(driver)

    # Combine existing URLs with new ones
    all_image_urls = existing_urls.union(image_links)
    print(f"Total unique image URLs: {len(all_image_urls)}")

    # Save the collected image URLs to a JSON file
    save_img_src_to_json(sorted(all_image_urls), json_path)

    print("Image collection complete.")