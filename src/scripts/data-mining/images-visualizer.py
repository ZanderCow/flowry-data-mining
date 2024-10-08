import os
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

# Define the path to the parent 'flowers' directory
flowers_dir = '../flowers'

# Initialize dictionaries to hold the analysis data
flower_counts = {}
flower_file_sizes = {}
flower_image_dimensions = {}

# Loop through each folder in the 'flowers' directory
for folder in os.listdir(flowers_dir):
    folder_path = os.path.join(flowers_dir, folder)
    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        flower_counts[folder] = len(image_files)
        
        # Initialize lists for file sizes and dimensions
        file_sizes = []
        dimensions = []
        
        # Loop through images to gather file size and dimensions
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            
            # Get file size in KB
            file_size = os.path.getsize(image_path) / 1024
            file_sizes.append(file_size)
            
            # Get image dimensions using PIL
            with Image.open(image_path) as img:
                width, height = img.size
                dimensions.append((width, height))
        
        flower_file_sizes[folder] = file_sizes
        flower_image_dimensions[folder] = dimensions

# Prepare a DataFrame to summarize statistics
data = {
    'Flower Category': [],
    'Number of Images': [],
    'Average File Size (KB)': [],
    'Average Width (px)': [],
    'Average Height (px)': [],
}

for flower, count in flower_counts.items():
    sizes_list = flower_file_sizes.get(flower, [])
    dimensions_list = flower_image_dimensions.get(flower, [])
    
    # Handle empty lists to avoid warnings and NaN values
    avg_size = np.mean(sizes_list) if sizes_list else 0
    avg_dimensions = np.mean(dimensions_list, axis=0) if dimensions_list else (0, 0)
    
    data['Flower Category'].append(flower)
    data['Number of Images'].append(count)
    data['Average File Size (KB)'].append(avg_size)
    data['Average Width (px)'].append(avg_dimensions[0])
    data['Average Height (px)'].append(avg_dimensions[1])

df = pd.DataFrame(data)

# Calculate the total number of images
total_images = df['Number of Images'].sum()

# Create the plots using Plotly

# First plot: Number of Images in Each Flower Category
bar1 = go.Bar(
    x=df['Flower Category'],
    y=df['Number of Images'],
    marker=dict(color='skyblue')
)

layout1 = go.Layout(
    title='Number of Images in Each Flower Category',
    xaxis=dict(title='Flower Category'),
    yaxis=dict(title='Number of Images'),
    margin=dict(b=150),
)

fig1 = go.Figure(data=[bar1], layout=layout1)
plot1_html = plot(fig1, include_plotlyjs=False, output_type='div')

# Second plot: Average File Size in Each Flower Category
bar2 = go.Bar(
    x=df['Flower Category'],
    y=df['Average File Size (KB)'],
    marker=dict(color='lightgreen')
)

layout2 = go.Layout(
    title='Average File Size in Each Flower Category',
    xaxis=dict(title='Flower Category'),
    yaxis=dict(title='Average File Size (KB)'),
    margin=dict(b=150),
)

fig2 = go.Figure(data=[bar2], layout=layout2)
plot2_html = plot(fig2, include_plotlyjs=False, output_type='div')

# Third plot: Average Image Dimensions for Each Flower Category
bar_width = go.Bar(
    x=df['Flower Category'],
    y=df['Average Width (px)'],
    name='Width',
    marker=dict(color='coral')
)

bar_height = go.Bar(
    x=df['Flower Category'],
    y=df['Average Height (px)'],
    name='Height',
    marker=dict(color='lightcoral')
)

layout3 = go.Layout(
    title='Average Image Dimensions for Each Flower Category',
    xaxis=dict(title='Flower Category'),
    yaxis=dict(title='Image Dimensions (pixels)'),
    barmode='group',
    margin=dict(b=150),
)

fig3 = go.Figure(data=[bar_width, bar_height], layout=layout3)
plot3_html = plot(fig3, include_plotlyjs=False, output_type='div')

# Create the HTML file to embed the table and plots
html_content = f"""
<html>
<head>
    <title>Flower Image Data Analysis</title>
    <!-- Include Plotly.js library -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .content-wrapper {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        table, th, td {{
            border: 1px solid black;
        }}
        th, td {{
            padding: 10px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        h2 {{
            text-align: center;
        }}
        .plot-container {{
            max-width: 100%;
            overflow-x: auto;
        }}
        .total-images {{
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="content-wrapper">
        <h1>Flower Image Data Analysis</h1>

        <div class="total-images">
            Total Number of Images: {total_images}
        </div>

        <h2>Summary Statistics</h2>
        <div class="table-container">
            {df.to_html(index=False)}
        </div>

        <h2>Number of Images in Each Flower Category</h2>
        <div class="plot-container">
            {plot1_html}
        </div>

        <h2>Average File Size in Each Flower Category</h2>
        <div class="plot-container">
            {plot2_html}
        </div>

        <h2>Average Image Dimensions in Each Flower Category</h2>
        <div class="plot-container">
            {plot3_html}
        </div>
    </div>
</body>
</html>
"""

# Save the HTML content to a file
with open('flower_image_data_analysis.html', 'w') as f:
    f.write(html_content)

print("HTML report generated: flower_image_data_analysis.html")