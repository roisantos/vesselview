import shutil
import torch
from torch.optim.lr_scheduler import CyclicLR
import datetime as dt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))

torch.cuda.empty_cache()  # Clear memory cache

from dataset import *
from utils import *
from settings_benchmark import *

from dataset import writer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate

# Función para manejar múltiples GPUs si están disponibles
def get_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            return torch.device('cuda'), num_gpus
        else:
            print("Using 1 GPU for training.")
            return torch.device('cuda:0'), 1
    else:
        print("Using CPU for training.")
        return torch.device('cpu'), 0

device, num_gpus = get_device()

# Obtener la fecha y hora actual para crear un directorio único
current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
root_result = f"test_runs/result_{current_time}"  # Nombre del directorio con la fecha y hora de ejecución

if not os.path.exists(root_result):
    os.makedirs(root_result)

# Definir una función collate personalizada
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:  # return an empty batch if all items are None
        return None
    return default_collate(batch)

# Cargar los datasets
all_dataset = prepareDatasets()
print(f"Models: {[name for name in models]}")
print(f"Datasets: {[name for name in all_dataset]}")

# Para cada modelo
for name_model in models:
    root_result_model = os.path.join(root_result, name_model)
    if not os.path.exists(root_result_model):
        os.makedirs(root_result_model)

    # Inicializar el modelo
    model: nn.Module = models[name_model]().to(device)

    # Si hay más de una GPU, usar DataParallel
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    register_hooks(model)

    # Para cada dataset
    for name_dataset in all_dataset:
        dataset = all_dataset[name_dataset]
        
        train_loader = DataLoader(dataset=dataset['train'], batch_size=8, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(dataset=dataset['val'], batch_size=4, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(dataset=dataset['test'], batch_size=4, shuffle=True)

        root_result_model_dataset = os.path.join(root_result_model, name_dataset)
        path_flag = os.path.join(root_result_model_dataset, f"finished.flag")

        if os.path.exists(path_flag):
            continue
        if os.path.exists(root_result_model_dataset):
            shutil.rmtree(root_result_model_dataset)
        os.makedirs(root_result_model_dataset)

        print(f"\n\n\nCurrent Model: {name_model}, Current training dataset: {name_dataset}")

        log_section = f"{name_model}_{name_dataset}"

        funcLoss = DiceLoss() if 'loss' not in dataset else dataset['loss']
        thresh_value = None if 'thresh' not in dataset else dataset['thresh']
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                     lr=1e-4, weight_decay=0.001)
        NUM_MAX_EPOCH = 300
        bestResult = {"epoch": -1, "dice": -1}
        ls_best_result = []

        for epoch in range(NUM_MAX_EPOCH):
            torch.cuda.empty_cache()

            log_section_parent = f"{log_section}"
            result_train = traverseDataset(model=model, loader=train_loader, epoch=epoch,
                                           thresh_value=thresh_value, 
                                           log_section=f"{log_section_parent}_{epoch}_train",
                                           log_writer=writer if epoch % 1 == 0 else None,
                                           description=f"Train Epoch {epoch}", device=device,
                                           funcLoss=funcLoss, optimizer=optimizer)

            for key in result_train:
                writer.add_scalar(tag=f"{log_section}/{key}_train", scalar_value=result_train[key], global_step=epoch)

            # Validación
            result = traverseDataset(model=model, loader=val_loader, epoch=epoch,
                                     thresh_value=thresh_value, 
                                     log_section=f"{log_section_parent}_{epoch}_val",
                                     log_writer=writer if epoch % 1 == 0 else None,
                                     description=f"Val Epoch {epoch}", device=device,
                                     funcLoss=funcLoss, optimizer=None)
            for key in result:
                writer.add_scalar(tag=f"{log_section}/{key}_val", scalar_value=result[key], global_step=epoch)

            # Evaluación en el test
            dice = result['dice']
            print(f"val dice: {dice}. ({name_model} on {name_dataset})")
            if dice > bestResult['dice']:
                bestResult['dice'] = dice
                bestResult['epoch'] = epoch
                ls_best_result.append(f"epoch={epoch}, val_dice={dice:.3f}")
                print("Best dice found. Evaluating on testset...")

                result = traverseDataset(model=model, loader=test_loader, epoch=epoch,
                                         thresh_value=thresh_value, 
                                         log_section=None, log_writer=None,
                                         description=f"Test Epoch {epoch}", device=device,
                                         funcLoss=funcLoss, optimizer=None)
                ls_best_result.append(result)

                path_json = os.path.join(root_result_model_dataset, "best_result.json")
                with open(path_json, "w") as f:
                    json.dump(ls_best_result, f, indent=2)
                path_model = os.path.join(root_result_model_dataset, 'model_best.pth')
                torch.save(model.state_dict(), path_model)
            else:
                threshold = 100
                if epoch - bestResult['epoch'] >= threshold:
                    print(f"Precision didn't improve in recent {threshold} epochs, stopping training.")
                    break

        with open(path_flag, "w") as f:
            f.write("training and testing finished.")
