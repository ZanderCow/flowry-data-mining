import json

# Define the file paths
json_file_path_1 = '../flowers/Rose_img_src_combined.json'
json_file_path_2 = '../flowers/Rose_img_src.json'
output_json_file_path = '../flowers/Rose_img_src_combined2.json'

# Read the first JSON file
with open(json_file_path_1, 'r') as file1:
    data1 = json.load(file1)

# Read the second JSON file
with open(json_file_path_2, 'r') as file2:
    data2 = json.load(file2)

# Combine the contents of the two JSON files
combined_data = data1 + data2

# Write the combined data to the output JSON file
with open(output_json_file_path, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)

print(f"Combined JSON data has been saved to {output_json_file_path}")