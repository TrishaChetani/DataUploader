import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Define the CustomDataset class for loading images
class CustomDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, categories=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.categories = categories if categories is not None else ['juice_bottle']
        self.image_files, self.labels = self._get_image_files()
        
    def _get_image_files(self):
        image_files = []
        labels = []
        for category in self.categories:
            category_dir = os.path.join(self.data_dir, category, self.split)
            if not os.path.exists(category_dir):
                continue
            for subdir in os.listdir(category_dir):
                subdir_path = os.path.join(category_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path) and file.lower().endswith('.png'):
                            image_files.append(file_path)
                            if 'good' in subdir:
                                labels.append(0)  # Good label
                            elif 'logical_anomalies' in subdir:
                                labels.append(1)  # Logical anomalies label
                            elif 'structural_anomalies' in subdir:
                                labels.append(2)  # Structural anomalies label
                            else:
                                labels.append(-1)  # Unknown category
        return image_files, labels
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define transformations (if applicable)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Directory paths
data_dir = '/home/TrishaChetani/mvtec_loco_anomaly_detection'
categories = ['pushpins', 'splicing_connectors', 'juice_bottle', 'screw_bag', 'breakfast_box']

# Verify base directory exists
if not os.path.exists(data_dir):
    print(f"Base directory does not exist: {data_dir}")
else:
    print(f"Base directory exists: {data_dir}")

# Create datasets for different splits
train_dataset = CustomDataset(data_dir=data_dir, split='train', transform=transform, categories=categories)
test_dataset = CustomDataset(data_dir=data_dir, split='test', transform=transform, categories=categories)
validate_dataset = CustomDataset(data_dir=data_dir, split='validation', transform=transform, categories=categories)

# Print the length of each dataset for debugging
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in test dataset: {len(test_dataset)}")
print(f"Number of samples in validate dataset: {len(validate_dataset)}")

# Create dataloaders if datasets are not empty
batch_size = 32

if len(train_dataset) > 0:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
else:
    print("Train dataset is empty. Skipping DataLoader creation for train.")

if len(test_dataset) > 0:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
else:
    print("Test dataset is empty. Skipping DataLoader creation for test.")

if len(validate_dataset) > 0:
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
else:
    print("Validate dataset is empty. Skipping DataLoader creation for validate.")

# Placeholder function for processing each DataLoader
def process_dataloader(dataloader, split):
    total_batch_process_time = 0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Example: Process each batch
        # You can modify this part based on your specific task
        if split == 'test':
            # Test split labeling (example)
            test_labels = [label.item() for label in labels]
            print(f"Test Labels: {test_labels}")
        elif split in ['train', 'validate']:
            # Train and validate split labeling (example)
            train_validate_labels = [label.item() for label in labels]
            print(f"Train/Validate Labels: {train_validate_labels}")

        # Example: Simulate processing time for each batch
        batch_start_time = time.time()

        # Example: Placeholder for actual processing (commented out for clarity)
        # print(images.size())  # Print batch size for debugging
        # print(labels)  # Print labels for debugging

        batch_end_time = time.time()
        batch_process_time = batch_end_time - batch_start_time
        total_batch_process_time += batch_process_time
        num_batches += 1
        print(f"Time to process batch {num_batches}: {batch_process_time:.2f} seconds")

    average_batch_process_time = total_batch_process_time / num_batches
    print(f"Average time to process a batch: {average_batch_process_time:.2f} seconds")

# Process each dataloader
print("Processing Train Loader")
if 'train_loader' in locals():
    process_dataloader(train_loader, 'train')

print("Processing Test Loader")
if 'test_loader' in locals():
    process_dataloader(test_loader, 'test')

print("Processing Validate Loader")
if 'validate_loader' in locals():
    process_dataloader(validate_loader, 'validate')
