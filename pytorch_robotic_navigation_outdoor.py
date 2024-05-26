# Terrapn: Unstructured Terrain Navigation with Online Terrain Segmentation and Adaptation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Define the OutdoorNavigationDataset class to handle data loading
class OutdoorNavigationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize the dataset
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get an item from the dataset at the given index
        image_path, label = self.data[idx]
        # Load the image
        image = Image.open(image_path).convert('RGB')
        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def load_data(self):
        # Load the data from the data directory
        # Assumes the data is organized in the following structure:
        # data_dir/
        #     images/
        #         image_0.jpg
        #         image_1.jpg
        #         ...
        #     labels.txt
        image_dir = os.path.join(self.data_dir, 'images')
        label_file = os.path.join(self.data_dir, 'labels.txt')
        # Read the label file and create a list of (image_path, label) tuples
        with open(label_file, 'r') as f:
            lines = f.readlines()
        data = [(os.path.join(image_dir, line.strip().split()[0]), int(line.strip().split()[1])) for line in lines]
        return data

# Define the TerrainSegmentationModel class
class TerrainSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(TerrainSegmentationModel, self).__init__()
        # Define the model architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass through the model
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Set the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the data directory
data_dir = 'path/to/your/data/directory'

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the OutdoorNavigationDataset
dataset = OutdoorNavigationDataset(data_dir, transform=transform)

# Create a DataLoader to handle batching and shuffling
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set the model hyperparameters
num_classes = 3  # Assumes 3 terrain classes (e.g., grass, gravel, asphalt)
learning_rate = 0.001
num_epochs = 10

# Create an instance of the TerrainSegmentationModel
model = TerrainSegmentationModel(num_classes).to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        # Move the data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Print the average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'terrain_segmentation_model.pth')

# Possible Errors and Solutions:
# 1. Error: FileNotFoundError when loading the dataset.
#    Solution: Ensure that the `data_dir` variable points to the correct directory containing the 'images' folder and
#              'labels.txt' file.
#
# 2. Error: ValueError: invalid literal for int() with base 10: 'label'.
#    Solution: Check the format of the 'labels.txt' file. Each line should contain the image filename and its corresponding
#              label separated by a space. Ensure that the labels are integers.
#
# 3. Error: RuntimeError: CUDA out of memory.
#    Solution: Reduce the batch size to fit the data within the available GPU memory. If the issue persists, consider
#              using a smaller model architecture or downsizing the input images.
#
# 4. Error: Poor segmentation performance.
#    Solution: Experiment with different model architectures, such as using deeper or more complex networks like U-Net or
#              DeepLab. Increase the size and diversity of the dataset, covering various terrain types and conditions.
#              Fine-tune the hyperparameters and consider using data augmentation techniques.

# Note: The code assumes a specific dataset format where the images are stored in the 'images' folder and the corresponding
#       labels are provided in the 'labels.txt' file. Adapt the `load_data` method according to your dataset's structure
#       and file formats.

# To use the trained model for outdoor navigation:
# 1. Load the saved model weights using `model.load_state_dict(torch.load('terrain_segmentation_model.pth'))`.
# 2. Acquire an image from the robot's camera or sensor.
# 3. Preprocess the image using the same transformations applied during training.
# 4. Pass the preprocessed image through the model to obtain the terrain segmentation predictions.
# 5. Interpret the segmentation predictions and use them for navigation decisions, such as avoiding obstacles or
#    adapting the robot's behavior based on the terrain type.
# 6. Repeat steps 2-5 in real-time as the robot navigates through the outdoor environment.
