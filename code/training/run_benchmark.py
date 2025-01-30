import os
import sys
import shutil
import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
import datetime as dt
import json
import argparse


# Clear CUDA cache
torch.cuda.empty_cache()

# Set up root directory and custom module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from ds.dataset import prepare_datasets_from_json
from utils.utils import *
#from config.settings_benchmark import models  # Assuming `models` is a dictionary with available models
from models.common import *
from models.frnet import * 
from models.roinet import * 

# Initialize SummaryWriter for TensorBoard
writer = SummaryWriter()

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def select_device_():
    """Selects CUDA device if available, otherwise CPU."""
    count_card = torch.cuda.device_count()
    id_card = 0
    if count_card > 1:
        while True:
            s = input(f"Choose video card number (0-{count_card-1}): ")
            if s.isdigit() and (0 <= int(s) < count_card):
                id_card = int(s)
                break
            print("Invalid input!")
    return torch.device(f'cuda:{id_card}' if torch.cuda.is_available() else 'cpu')

def select_device():
    """Selects CUDA device if available, otherwise CPU."""
    count_card = torch.cuda.device_count()
    id_card = 0
    if count_card > 1:
        print(f"Using {count_card} GPUs")  # Imprimir el número de GPUs disponibles
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate(batch):
    """Custom collate function to skip empty batches."""
    # Debugging: Log raw batch contents
    # print(f"Raw batch contents: {batch}")

    # Filter out None values
    batch = [item for item in batch if item is not None]

    # Debugging: Check if batch is empty
    # if len(batch) == 0:
    #     print("##### BATCH IS EMPTY #####")

    return None if len(batch) == 0 else default_collate(batch)

    
def load_models_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    models = {}
    for name, model_config in config["models"].items():
        if "RoiNet" in model_config["type"]:
            models[name] = lambda: RoiNet(
                ch_in=model_config.get("ch_in", 3),
                ch_out=model_config.get("ch_out", 1),
                ls_mid_ch=model_config.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                cls_init_block=eval(model_config.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(model_config.get("cls_conv_block", "ResidualBlock"))
            )
        elif "FRNet" in model_config["type"]:
            models[name] = lambda: RoiNet(
                ch_in=model_config.get("ch_in", 3),
                ch_out=model_config.get("ch_out", 1),
                cls_init_block=eval(model_config.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(model_config.get("cls_conv_block", "ResidualBlock"))
            )
    return models

# ---------------------------------------
# TRAINING AND EVALUATION FUNCTION
# ---------------------------------------
def train_and_evaluate(model_name, dataset, config, logging_enabled=False):
    """
    Trains and evaluates a specific model on a dataset.
    Saves the best model and results based on dice score.
    """
    device = select_device()
    model: torch.nn.Module = models[model_name]().to(device)

    print("\nModelo cargado en GPU:")
    print(f"- Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(f"- Memoria ocupada por modelo en GPU: {sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 2):.2f} MB")
    print_gpu_memory_info("Después de cargar el modelo en GPU")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    # Get training parameters from config
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 300)
    thresh_value = training_config.get("early_stopping_threshold", 100)
    batch_size = training_config.get("batch_size", 8)
    num_workers = training_config.get("num_workers", 32)
    learning_rate = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 0.001)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr=learning_rate, weight_decay=weight_decay
    )
    funcLoss = DiceLoss() if 'loss' not in dataset else dataset['loss']

    # Configure DataLoaders
    trainLoader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=custom_collate,num_workers=num_workers, pin_memory=True)
    if(batch_size%2==0):
        valLoader = DataLoader(dataset=dataset['val'], batch_size=batch_size // 2, shuffle=True, collate_fn=custom_collate,num_workers=num_workers, pin_memory=True)
    else:
        valLoader = DataLoader(dataset=dataset['val'], batch_size=1, shuffle=True, collate_fn=custom_collate,num_workers=num_workers, pin_memory=True)

    testLoader = DataLoader(dataset=dataset['test'])

    bestResult = {"epoch": -1, "dice": -1}
    ls_best_result = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # Training
        result_train = traverseDataset(
            model=model, loader=trainLoader, epoch=epoch, thresh_value=thresh_value,
            log_section=f"{model_name}_{epoch}_train", log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Train Epoch {epoch}", device=device, funcLoss=funcLoss, optimizer=optimizer
        )

        # Log training metrics
        for key, value in result_train.items():
            writer.add_scalar(f"{model_name}/{key}_train", value, epoch)

        # Validation
        result_val = traverseDataset(
            model=model, loader=valLoader, epoch=epoch, thresh_value=thresh_value,
            log_section=f"{model_name}_{epoch}_val", log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Val Epoch {epoch}", device=device, funcLoss=funcLoss
        )

        # Log validation metrics
        for key, value in result_val.items():
            writer.add_scalar(f"{model_name}/{key}_val", value, epoch)

        # Evaluate dice score and update if it's the best model so far
        dice = result_val['dice']
        print(f"Validation Dice: {dice} for Model: {model_name}")

        if dice > bestResult['dice']:
            bestResult.update({"epoch": epoch, "dice": dice})
            ls_best_result.append({"epoch": epoch, "val_dice": dice})
            print("New best dice found, evaluating on test set...")

            result_test = traverseDataset(
                model=model, loader=testLoader, epoch=epoch, thresh_value=thresh_value,
                log_section=None, log_writer=None, description=f"Test Epoch {epoch}", device=device, funcLoss=funcLoss
            )
            ls_best_result.append(result_test)
            save_best_results(model, ls_best_result, model_name)

        # Early stopping
        if epoch - bestResult['epoch'] >= thresh_value:
            print(f"Stopping training: no improvement in last {thresh_value} epochs.")
            break

def train_and_evaluate_old(model_name, dataset, epochs=300, threshold=300, logging_enabled=False):
    """
    Trains and evaluates a specific model on a dataset.
    Saves the best model and results based on dice score.
    """
    device = select_device()
    model: torch.nn.Module = models[model_name]().to(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr=1e-4, weight_decay=0.001
    )
    funcLoss = DiceLoss() if 'loss' not in dataset else dataset['loss']
    thresh_value = dataset.get('thresh')

    # Configure DataLoaders
    trainLoader = DataLoader(dataset=dataset['train'], batch_size=8, shuffle=True, collate_fn=custom_collate)
    print("TrainLoader: ",trainLoader)
    valLoader = DataLoader(dataset=dataset['val'], batch_size=4, shuffle=True, collate_fn=custom_collate)
    testLoader = DataLoader(dataset=dataset['test'])

    bestResult = {"epoch": -1, "dice": -1}
    ls_best_result = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # Training
        print("debug1")
        result_train = traverseDataset(
            model=model, loader=trainLoader, epoch=epoch, thresh_value=thresh_value,
            log_section=f"{model_name}_{epoch}_train", log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Train Epoch {epoch}", device=device, funcLoss=funcLoss, optimizer=optimizer
        )

        # Log training metrics
        for key, value in result_train.items():
            writer.add_scalar(f"{model_name}/{key}_train", value, epoch)

        # Validation
        result_val = traverseDataset(
            model=model, loader=valLoader, epoch=epoch, thresh_value=thresh_value,
            log_section=f"{model_name}_{epoch}_val", log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Val Epoch {epoch}", device=device, funcLoss=funcLoss
        )

        # Log validation metrics
        for key, value in result_val.items():
            writer.add_scalar(f"{model_name}/{key}_val", value, epoch)

        # Evaluate dice score and update if it's the best model so far
        dice = result_val['dice']
        print(f"Validation Dice: {dice} for Model: {model_name}")

        if dice > bestResult['dice']:
            bestResult.update({"epoch": epoch, "dice": dice})
            ls_best_result.append({"epoch": epoch, "val_dice": dice})
            print("New best dice found, evaluating on test set...")

            result_test = traverseDataset(
                model=model, loader=testLoader, epoch=epoch, thresh_value=thresh_value,
                log_section=None, log_writer=None, description=f"Test Epoch {epoch}", device=device, funcLoss=funcLoss
            )
            ls_best_result.append(result_test)
            save_best_results(model, ls_best_result, model_name)

        # Early stopping
        if epoch - bestResult['epoch'] >= threshold:
            print(f"Stopping training: no improvement in last {threshold} epochs.")
            break

# ---------------------------------------
# SAVE RESULTS FUNCTION
# ---------------------------------------

def save_best_results(model, results, model_name):
    """Saves the best model and results in a unique timestamped directory."""
    root_result = f"run_benchmark_runs/result_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{model_name}"
    os.makedirs(root_result, exist_ok=True)

    # Save results and model
    with open(os.path.join(root_result, "best_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(root_result, "model_best.pth"))
    with open(os.path.join(root_result, "finished.flag"), "w") as f:
        f.write("training and testing finished.")

# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run benchmark for a specific model")
    parser.add_argument("-model", type=str, required=True, help="Name of the model to train")
    args = parser.parse_args()

    # Load configuration from JSON file
    config_path = 'code/config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Print configuration before starting
    print("Loaded configuration:")
    print(json.dumps(config, indent=4))

    logging_enabled = config["training"].get("logging_enabled", True)  

    models = load_models_from_json(config_path)
    all_datasets = prepare_datasets_from_json(config_path)
    #all_datasets = prepareDatasets()

    print(f"Available Models: {[name for name in models]}")
    print(f"Available Datasets: {[name for name in all_datasets]}")
    # Check if the requested model exists
    model_name = args.model
    if model_name not in models:
        print(f"Error: Model '{model_name}' not found in the configuration.")
        sys.exit(1)

    print(f"\n\nTraining Model: {model_name}")
    for dataset_name, dataset in all_datasets.items():
        print(f"Using Dataset: {dataset_name}")
        train_and_evaluate(model_name, dataset, config, logging_enabled=logging_enabled)
    # # Iterate over each model and each dataset
    # for name_model in models:
    #     print(f"\n\nTesting Model: {name_model}")

    #     for dataset_name, dataset in all_datasets.items():
    #         print(f"Current Model: {name_model} with Dataset: {dataset_name}")
    #         train_and_evaluate(name_model, dataset, config, logging_enabled=logging_enabled)
