import sys
import os

# Set up root directory and custom module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from ds.dataset import SegmentationDataset

# Initialize the dataset
dataset_path = "./dataset/FIVES"
dataset = SegmentationDataset(dataset_path)

# Test the dataset
for i in range(5):  # Adjust the range as needed
    try:
        name, image, label = dataset[i]
        print(f"Sample {i}: Name={name}, Image Shape={image.shape}, Label Shape={label.shape}")
    except Exception as e:
        print(f"Error at index {i}: {e}")
