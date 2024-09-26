# Flower Data Mining

This repository contains the data miner for my Flowry app. The data miner is designed to efficiently gather and process data related to various flowers.

## Features

- Efficient data collection
- Data processing and analysis
- Integration with the Flowry app

## Installation

1. Install the necessary Python dependencies by running:

    ```bash
    python3 -m venv myenv
    ```

    ```bash
    pip install -r requirements.txt
    ```

2. Ensure that **Docker** is installed on your system. You can install Docker by following the instructions [here](https://docs.docker.com/get-docker/).

3. Run the following command to start the Chrome WebDriver through Docker:

    ```bash
    docker run -d -p 4444:4444 -p 5900:5900 -p 7900:7900 --shm-size=2g selenium/standalone-chromium:latest
    ```

## Usage

To start the data miner, use the following command:

```bash
python data_miner.py