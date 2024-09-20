import os
from jinja2 import Template

# Define the root directory containing all flower folders
root_dir = 'flowers'

# Create a list of all images in subdirectories with their names
images = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(subdir, file)
            image_name = os.path.basename(file)  # Extract just the filename
            images.append({'path': image_path, 'name': image_name})

# Create a basic HTML template using Jinja2
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flower Image Gallery</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(50px, 1fr));
            gap: 5px;
            max-width: 1200px;
            margin: auto;
        }
        .grid-container img {
            width: 100%;
            height: auto;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <div class="grid-container">
        {% for image in images %}
        <img src="{{ image.path }}" alt="Flower Image" title="{{ image.name }}">
        {% endfor %}
    </div>
</body>
</html>
"""

# Render the template with the list of images and their names
template = Template(html_template)
rendered_html = template.render(images=images)

# Save the rendered HTML to a file
with open('gallery.html', 'w') as f:
    f.write(rendered_html)

print("HTML gallery generated successfully! Open 'gallery.html' to view it.")