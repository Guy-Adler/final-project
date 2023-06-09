from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from ..config import BATCH_SIZE, BASE_PATH

# Get the data folder paths
train_dir = os.path.join(BASE_PATH, 'train', '')
test_dir = os.path.join(BASE_PATH, 'test', '')

# Configure the transformations the data will go through after being loaded
transformations = transforms.Compose([
    transforms.Grayscale(),  # Convert the images to 1 color channel
    transforms.ToTensor(),  # Convert the images from a PIL image to a tensor
])

# Load the train and test datasets:
train_dataset = datasets.ImageFolder(root=train_dir, transform=transformations)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transformations)

# Load the datasets into dataloaders, which provide a nice iterable interface with batching
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Expose some data loader properties for ease of access
class_to_idx = test_dataset.class_to_idx  # Class to index mappings

# The character '<' is mapped as "arrow", because '<' can't be a folder name.
# Map it back to the original '<' for usage.
idx_to_class = {value: key if key != 'arrow' else '<' for key, value in class_to_idx.items()}  # Index to class mappings
classes = list(idx_to_class.values())  # Class names
