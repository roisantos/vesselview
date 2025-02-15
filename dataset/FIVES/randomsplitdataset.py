import os
import shutil
import random

# Define paths
image_dir = "train/image"
label_dir = "train/label"
val_image_dir = "val/image"
val_label_dir = "val/label"

# Ensure validation directories exist
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

# Ensure images and labels match
image_files = [f for f in image_files if os.path.splitext(f)[0] in [os.path.splitext(l)[0] for l in label_files]]
label_files = [f for f in label_files if os.path.splitext(f)[0] in [os.path.splitext(i)[0] for i in image_files]]

# Shuffle files
random.seed(42)  # For reproducibility
combined = list(zip(image_files, label_files))
random.shuffle(combined)

# Split dataset
split_index = int(len(combined) * 0.8)
train_set = combined[:split_index]
val_set = combined[split_index:]

# Move validation files
for img_file, lbl_file in val_set:
    shutil.move(os.path.join(image_dir, img_file), os.path.join(val_image_dir, img_file))
    shutil.move(os.path.join(label_dir, lbl_file), os.path.join(val_label_dir, lbl_file))

print("Dataset split completed.")
