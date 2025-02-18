import os
import sys
import cv2
import json
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import *
from models.common import *
from utils.utils import *
from training.loss import *
from tqdm import tqdm

# Set up paths and logging directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
log_root = f"tensorboardLog/{timestamp}"
writer = SummaryWriter(log_dir=log_root)

# Load config
config_path = os.path.join(ROOT_DIR, 'config/config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
dataset_base_path = config["datasets"]["base_path"]

# --- Optional: Load Restormer Model for Preprocessing ---
try:
    from restormer import Restormer  # Adjust import as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restormer_model = Restormer(pretrained=True).to(device)
    restormer_model.eval()
except ImportError:
    restormer_model = None
    print("Restormer model not found. Restormer preprocessing will be disabled.")

def apply_restormer(image: np.ndarray) -> np.ndarray:
    """
    Applies the Restormer model to the input image for deblurring/sharpening.
    Args:
        image (np.ndarray): Input image in BGR format (uint8).
    Returns:
        np.ndarray: The restored image in BGR format (uint8).
    """
    # Convert image to float32 and scale to [0,1]
    image_float = image.astype(np.float32) / 255.0
    # Convert BGR to RGB (if your Restormer expects RGB)
    image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
    # Convert to tensor and add batch dimension: (1, 3, H, W)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored_tensor = restormer_model(image_tensor)
    
    # Remove batch dimension and convert back to numpy (H, W, C)
    restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Convert from [0,1] to [0,255] and to uint8
    restored_img = np.clip(restored_img * 255.0, 0, 255).astype(np.uint8)
    # Convert RGB back to BGR
    restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
    return restored_img_bgr

# -----------------------------------------------------------------------------
# Functions to Prepare Datasets
# -----------------------------------------------------------------------------
def prepareDatasets():
    all_datasets = {}
    all_datasets['FIVES'] = {
        "train": SegmentationDataset(os.path.join(dataset_base_path, "train")),
        "test": SegmentationDataset(os.path.join(dataset_base_path, "test")),
        "val": SegmentationDataset(os.path.join(dataset_base_path, "val"))
    }
    return all_datasets

def prepare_datasets_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    datasets = {}
    base_path = config["datasets"]["base_path"]
    print(f"Base path from config: {base_path}")

    for name, ds_config in config["datasets"].items():
        if name == "base_path":
            continue

        paths = ds_config["paths"]
        batch_size = ds_config.get("batch_size", 8)
        augmentation_config = ds_config.get("augmentation", {})

        datasets[name] = {
            "train": SegmentationDataset(
                os.path.join(base_path, paths["train"]),
                augmentation_config=augmentation_config,
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                restormer=ds_config.get("preprocessing", {}).get("restormer", False)
            ),
            "val": SegmentationDataset(
                os.path.join(base_path, paths["val"]),
                augmentation_config=augmentation_config,
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                restormer=ds_config.get("preprocessing", {}).get("restormer", False)
            ),
            "test": SegmentationDataset(
                os.path.join(base_path, paths["test"]),
                augmentation_config=augmentation_config,
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                restormer=ds_config.get("preprocessing", {}).get("restormer", False)
            )
        }

        # Debug: Test a few samples (optional)
        train_dataset = datasets[name]["train"]
        print(f"Testing dataset: {name} (train split)")
        for i in range(20):
            try:
                name_sample, image_sample, label_sample = train_dataset[i]
                print(f"Sample {i}: Name={name_sample}, Image Shape={image_sample.shape}, Label Shape={label_sample.shape}")
            except Exception as e:
                print(f"Error at index {i}: {e}")

    return datasets

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, dataset_paths: Union[str, List[str]], augmentation_config=None, start: float = 0, end: float = 1, restormer: bool = False) -> None:
        """
        Dataset class for segmentation tasks.
        Args:
            dataset_paths (str or list): Path(s) to dataset directories.
            augmentation_config (dict): Augmentation configuration.
            start (float): Starting proportion.
            end (float): Ending proportion.
            restormer (bool): If True, apply Restormer preprocessing to each image.
        """
        super().__init__()
        self.ls_item = []
        self.augmentation_config = augmentation_config if augmentation_config is not None else {}
        self.restormer = restormer

        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        for path_dataset in dataset_paths:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")
            print(f"Absolute Dataset path: {os.path.abspath(path_dataset)}")
            print(f"Image dir: {os.path.abspath(path_dir_image)}")
            print(f"Label dir: {os.path.abspath(path_dir_label)}")

            if not os.path.exists(path_dir_image) or not os.path.exists(path_dir_label):
                print(f"Error: Missing image or label directory in {path_dataset}")
                continue

            valid_extensions = ('.png', '.jpg', '.jpeg')
            ls_image_files = [f for f in os.listdir(path_dir_image) if f.endswith(valid_extensions)]
            ls_label_files = [f for f in os.listdir(path_dir_label) if f.endswith(valid_extensions)]

            for name in ls_image_files:
                if name in ls_label_files:
                    path_image = os.path.join(path_dir_image, name)
                    path_label = os.path.join(path_dir_label, name)
                    assert os.path.exists(path_image), f"Image file does not exist: {path_image}"
                    assert os.path.exists(path_label), f"Label file does not exist: {path_label}"
                    self.ls_item.append({"name": name, "path_image": path_image, "path_label": path_label})

        if not self.ls_item:
            raise ValueError("Error: No valid images found in dataset.")

        random.seed(0)
        random.shuffle(self.ls_item)
        start_idx, end_idx = int(start * len(self.ls_item)), int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start_idx:end_idx]

    def __len__(self) -> int:
        return len(self.ls_item)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, np.ndarray]:
        index %= len(self)
        item = self.ls_item[index]
        name = item['name']
        image = cv2.imread(item['path_image'], cv2.IMREAD_COLOR)
        label = cv2.imread(item['path_label'], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Image failed to load at path: {item['path_image']}")
        if label is None:
            raise ValueError(f"Label failed to load at path: {item['path_label']}")

        # --- Optionally apply Restormer preprocessing ---
        if self.restormer and restormer_model is not None:
            try:
                image = apply_restormer(image)
            except Exception as e:
                print(f"Restormer preprocessing failed for {name}: {e}")

        # Apply augmentations if enabled
        if self.augmentation_config.get("enabled", False):
            image, label = self.augment(image, label)

        _, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)
        image, label = self.preprocess_image_label(image, label)
        return name, image, label

    def augment(self, image, label):
        # (Existing augmentation code here...)
        # [Refer to your current augment() implementation]
        return image, label

    def preprocess_image_label(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = image.astype("float32") / 255.0
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)

        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]
        return image, label