# Use the official Selenium standalone-chromium image as the base image
FROM selenium/standalone-chromium:latest

# Set the container name
LABEL container_name="selenium-chromium"

# Expose the necessary ports
EXPOSE 4444
EXPOSE 5900
EXPOSE 7900
