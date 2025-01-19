import os
import sys
import cv2
import json

# Set up paths and logging directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

import random
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import *
from models.common import *
from utils.utils import *
from training.loss import *
from tqdm import tqdm

# Prepare logging with a timestamp
timestamp = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
log_root = f"log/{timestamp}"
writer = SummaryWriter(log_dir=log_root)

# Dataset path for FIVES512
path_FIVES = os.path.join("dataset", "FIVES512")
assert os.path.exists(path_FIVES), f"Dataset path does not exist: {path_FIVES}"

# Functions
# -----------------------------------------------------------------------------

def prepareDatasets():
    all_datasets = {}
    all_datasets['FIVES512'] = {
        "train":SegmentationDataset(os.path.join(path_FIVES,"train"), ),
        "test":SegmentationDataset(os.path.join(path_FIVES, "test")),
        "val":SegmentationDataset(os.path.join(path_FIVES, "val"))
    }
    # all_datasets['FIVES1024'] = {
    #     "train":SegmentationDataset(os.path.join(path_FIVES,"train"), ),
    #     "test":SegmentationDataset(os.path.join(path_FIVES, "test")),
    #     "val":SegmentationDataset(os.path.join(path_FIVES, "val"))
    # }
    # all_datasets['DRIVE'] = {
    #     "train":SegmentationDataset(os.path.join(path_DRIVE,"train"), ),
    #     "test":SegmentationDataset(os.path.join(path_DRIVE, "test")),
    #     "val":SegmentationDataset(os.path.join(path_DRIVE, "val"))
    # }
    # all_datasets['OCTA500_6M'] = {
    #     "train":SegmentationDataset(os.path.join(path_OCTA500_6M,"train"), ),
    #     "val":SegmentationDataset(os.path.join(path_OCTA500_6M, "val")),
    #     "test":SegmentationDataset(os.path.join(path_OCTA500_6M,"test"))
    # }
    # all_datasets['OCTA500_3M'] = {
    #     "train":SegmentationDataset(os.path.join(path_OCTA500_3M,"train"), ),
    #     "val":SegmentationDataset(os.path.join(path_OCTA500_3M, "val")),
    #     "test":SegmentationDataset(os.path.join(path_OCTA500_3M,"test"))
    # }
    # all_datasets['ROSSA'] = {
    #     "train":SegmentationDataset([os.path.join(path_ROSSA, x) for x in ["train_manual", "train_sam"]], ),
    #     "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
    #     "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    # }
    # More datasets can be added here......
    
    return all_datasets
    
def prepare_datasets_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    datasets = {}
    for name, ds_config in config["datasets"].items():
        paths = ds_config["paths"]
        batch_size = ds_config.get("batch_size", 8)  # Default to 8 if not specified

        datasets[name] = {
            "train": DataLoader(SegmentationDataset(paths["train"], **ds_config.get("preprocessing", {})), batch_size=batch_size, shuffle=True),
            "val": DataLoader(SegmentationDataset(paths["val"], **ds_config.get("preprocessing", {})), batch_size=batch_size, shuffle=True),
            "test": DataLoader(SegmentationDataset(paths["test"], **ds_config.get("preprocessing", {})), batch_size=batch_size, shuffle=True)
        }
    return datasets

    


# Dataset Class
# -----------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, dataset_paths: Union[str, List[str]], start: float = 0, end: float = 1) -> None:
        """
        Dataset class for segmentation tasks. Loads images and labels from given paths.
        
        Args:
            dataset_paths (str or list): Path(s) to dataset directories.
            start (float): Starting proportion of the dataset to include (default is 0).
            end (float): Ending proportion of the dataset to include (default is 1).
        """
        super().__init__()
        self.ls_item = []
        
        # Ensure dataset_paths is a list
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        # Load dataset items from paths
        for path_dataset in dataset_paths:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")
            
            # Verify image and label directories exist
            if not os.path.exists(path_dir_image) or not os.path.exists(path_dir_label):
                print(f"Error: Missing image or label directory in {path_dataset}")
                continue

            # Load images and labels with specific file extensions
            valid_extensions = ('.png', '.jpg', '.jpeg')
            ls_image_files = [f for f in os.listdir(path_dir_image) if f.endswith(valid_extensions)]
            ls_label_files = [f for f in os.listdir(path_dir_label) if f.endswith(valid_extensions)]
            
            # Match images with labels
            for name in ls_image_files:
                if name in ls_label_files:
                    path_image = os.path.join(path_dir_image, name)
                    path_label = os.path.join(path_dir_label, name)
                    assert os.path.exists(path_image), f"Image file does not exist: {path_image}"
                    assert os.path.exists(path_label), f"Label file does not exist: {path_label}"
                    self.ls_item.append({"name": name, "path_image": path_image, "path_label": path_label})

        # Check for valid images
        if not self.ls_item:
            raise ValueError("Error: No valid images found in dataset.")

        # Shuffle and slice the dataset
        random.seed(0)
        random.shuffle(self.ls_item)
        start_idx, end_idx = int(start * len(self.ls_item)), int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start_idx:end_idx]

    def __len__(self) -> int:
        return len(self.ls_item)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Retrieves a dataset item at the specified index.
        
        Args:
            index (int): Index of the item to retrieve.
            
        Returns:
            tuple: (name, image, label) where image and label are preprocessed numpy arrays.
        """
        index %= len(self)
        item = self.ls_item[index]

        # Load and preprocess image and label
        name = item['name']
        image = cv2.imread(item['path_image'], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item['path_label'], cv2.IMREAD_GRAYSCALE)

        # Validate that images loaded correctly
        if image is None:
            print(f"Error: Image failed to load at path: {item['path_image']}")
            return None
        if label is None:
            print(f"Error: Label failed to load at path: {item['path_label']}")
            return None

        # Apply data augmentations with 50% probability
        if np.random.rand() > 0.5:
            image, label = self.apply_augmentation(image, label)

        # Threshold label to ensure binary values
        _, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)

        # Preprocess image and label
        image, label = self.preprocess_image_label(image, label)

        return name, image, label

    # Helper Methods
    # -----------------------------------------------------------------------------

    def apply_augmentation(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies random augmentations to image and label."""
        angle = np.random.uniform(-180, 180)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, rot_mat, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)
        contrast = np.random.uniform(0.75, 1.25)
        brightness = np.random.randint(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return image, label

    def preprocess_image_label(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Processes image and label for model input, including padding and normalization."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype("float32") / 255.0

        # Ensure dimensions are multiples of 32 for model compatibility
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)

        # Reshape for model compatibility
        image = image.reshape((1, image.shape[0], image.shape[1]))
        label = label.reshape((1, label.shape[0], label.shape[1]))

        return image, label
