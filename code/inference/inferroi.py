import cv2
import torch
import os
import sys
import numpy as np
import time
import csv

# Add directories to sys.path (adjust these as needed for your project structure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))
# Also add the parent directory ("code") of the current file ("code/inference")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision.utils import save_image
from models.roinet import RoiNet  # Import the RoiNet model
from models.common import *        # Ensure ResidualBlock and others are available

def compute_dice(pred, gt, eps=1e-6):
    """Compute Dice coefficient given binary numpy arrays for prediction and ground truth."""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + eps)

def run_inference_on_directory(image_dir, label_dir, output_dir, model_path):
    # Instantiate the model with 3 input channels (matching training)
    model = RoiNet(ch_in=3, ch_out=1, k_size=9, out_k_size=25)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare lists to accumulate per-image inference times and Dice scores,
    # and a list of dictionaries for CSV writing.
    inference_times = []
    dice_scores = []
    results_for_csv = []

    # Process each image file in the input directory
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)  # Assumes same filename in label_dir

        # Load the image in color (BGR, as in training)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Skipping file due to load failure: {image_path}")
            continue

        # Normalize the image to [0, 1]
        image = image.astype("float32") / 255.0

        # Pad the image so its dimensions are multiples of 32 (matching training)
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image_padded = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2,
                                          pad_x // 2, pad_x // 2,
                                          cv2.BORDER_CONSTANT, value=0)

        # Adjust image shape as in training: (H, W, C) -> (C, H, W)
        image_transposed = np.transpose(image_padded, (2, 0, 1))
        # Convert to a torch tensor and add a batch dimension: (1, C, H, W)
        image_tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(device)

        # Measure inference time
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image_tensor)
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)

        # Print output statistics for debugging
        print(f"{filename} - Inference Time: {inference_time*1000:.2f} ms | "
              f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")

        # Save the raw output
        base_name = os.path.splitext(filename)[0]
        tensor_save_path = os.path.join(output_dir, f"{base_name}_output_tensor.pth")
        image_save_path = os.path.join(output_dir, f"{base_name}_output_image.png")
        torch.save(output.cpu(), tensor_save_path)
        save_image(output, image_save_path)

        # Load the corresponding ground-truth label
        gt_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if gt_label is None:
            print(f"Warning: Could not load ground truth label for {filename}")
            continue

        # Threshold ground truth to binary (0, 1)
        _, gt_label = cv2.threshold(gt_label, 127, 1, cv2.THRESH_BINARY)

        # Pad the ground truth label to match the network's input size
        pad_x_label = (gt_label.shape[1] // 32 + 1) * 32 - gt_label.shape[1]
        pad_y_label = (gt_label.shape[0] // 32 + 1) * 32 - gt_label.shape[0]
        gt_label_padded = cv2.copyMakeBorder(gt_label, pad_y_label // 2, pad_y_label // 2,
                                             pad_x_label // 2, pad_x_label // 2,
                                             cv2.BORDER_CONSTANT, value=0)

        # Convert predicted output to binary mask using a threshold of 0.5
        pred_prob = output.cpu().numpy()[0, 0]  # shape: (H, W)
        pred_binary = (pred_prob > 0.5).astype(np.float32)

        # Compute Dice score
        dice = compute_dice(pred_binary, gt_label_padded.astype(np.float32))
        dice_scores.append(dice)

        print(f"{filename} - Dice Score: {dice:.4f}")

        # Append the results for this image to our list for CSV
        results_for_csv.append({
            "filename": filename,
            "inference_time_ms": inference_time * 1000,  # Convert seconds to ms
            "dice_score": dice
        })

    # Compute and print average inference time and Dice score
    avg_time = np.mean(inference_times) * 1000  # in milliseconds
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"\nAverage Inference Time: {avg_time:.2f} ms")
    print(f"Average Dice Score: {avg_dice:.4f}")

    # Write the per-image results into a CSV file
    csv_file_path = os.path.join(output_dir, "results.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["filename", "inference_time_ms", "dice_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_for_csv:
            writer.writerow(row)
    print(f"Results written to {csv_file_path}")

# ------------------ User Settings ------------------
# Directory containing the input images
image_dir = r"../dataset/FIVES/test/image"
# Directory containing the corresponding ground-truth labels
label_dir = r"../dataset/FIVES/test/label"
# Directory where the inference results will be saved
output_dir = os.path.join('inference_results', 'RoiNet_inference')
# Path to the trained RoiNet model weights (update this if needed)
model_path = '../bestmodel/model_best.pth'

# Run the inference
run_inference_on_directory(image_dir, label_dir, output_dir, model_path)
