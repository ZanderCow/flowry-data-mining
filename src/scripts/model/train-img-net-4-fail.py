# Import necessary libraries
import os
import zipfile
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Mount Google Drive if using Colab (optional)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not using Google Colab. Proceeding without mounting Google Drive.")

# Specify the path to your zip file in Google Drive or local file system
# Replace with your actual path
data_zip = '/content/drive/MyDrive/flowers.zip'  # For Google Colab
# data_zip = 'flowers.zip'  # For local environment
data_dir = 'flowers'  # The directory after unzipping

# Check if the dataset directory exists
if not os.path.exists(data_dir):
    if os.path.exists(data_zip):
        print(f"Found zip file at '{data_zip}'. Extracting...")
        # Unzip the dataset
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"Dataset unzipped to '{data_dir}' directory.")
    else:
        raise FileNotFoundError(f"'{data_zip}' not found. Please ensure the path is correct.")
else:
    print(f"Dataset directory '{data_dir}' already exists.")

# Ensure the dataset is split into train and val directories
# If not, split the dataset
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    # Split the dataset into train and val
    dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Save the splits to disk
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Copy images to train and val directories
    import shutil

    def copy_images(dataset, destination):
        for idx in range(len(dataset)):
            img_path, label = dataset.dataset.samples[dataset.indices[idx]]
            class_name = dataset.dataset.classes[label]
            dest_dir = os.path.join(destination, class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy(img_path, dest_dir)

    print("Splitting dataset into train and val directories...")
    copy_images(train_dataset, train_dir)
    copy_images(val_dataset, val_dir)
    print("Dataset split into train and val directories.")
else:
    print("Train and val directories already exist.")

# Set parameters
batch_size = 16  # Adjusted to avoid potential memory issues
img_height = 224  # Adjusted to match model input size
img_width = 224
num_classes = 15  # Update based on your dataset
epochs = 10
learning_rate = 1e-4

# Check for corrupted images and remove them
from PIL import Image

def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        print(f"Corrupted image: {image_path}, error: {e}")
        return False

for dataset_dir in [train_dir, val_dir]:
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_image_valid(file_path):
                os.remove(file_path)
                print(f"Removed corrupted image: {file_path}")

# Data augmentation and normalization using timm's transforms
data_config = timm.data.resolve_data_config({}, model='mobilenetv4_conv_aa_medium')
train_transforms = timm.data.create_transform(**data_config, is_training=True)
val_transforms = timm.data.create_transform(**data_config, is_training=False)

# Create datasets
train_dataset = datasets.ImageFolder(train_dir, train_transforms)
val_dataset = datasets.ImageFolder(val_dir, val_transforms)

# Create data loaders with num_workers=0 to prevent potential deadlocks
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Print class names
class_names = train_dataset.classes
print(f'Class names: {class_names}')
num_classes = len(class_names)

# Load the pre-trained model from timm
model_name = 'mobilenetv4_conv_aa_medium'
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model.to(device)

# Optionally, freeze some layers
for name, param in model.named_parameters():
    if 'classifier' not in name and 'fc' not in name:
        param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Training and validation loops
best_accuracy = 0.0
epoch_times = []  # List to store epoch durations

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        print(f"Starting phase: {phase}")
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model if it has better accuracy
        if phase == 'val' and epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_wts = model.state_dict()

    # Calculate and display epoch duration
    epoch_duration = time.time() - epoch_start_time
    epoch_times.append(epoch_duration)
    print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds\n")

# Calculate total training time and average time per epoch
total_training_time = sum(epoch_times)
average_epoch_time = total_training_time / len(epoch_times)
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {average_epoch_time:.2f} seconds")

print(f'Best validation accuracy: {best_accuracy:.4f}')

# Load best model weights
model.load_state_dict(best_model_wts)

# Save the model
torch.save(model.state_dict(), 'flower_classifier_timm.pth')
print('Model saved as flower_classifier_timm.pth')