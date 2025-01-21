import torch
from torch.utils.data import DataLoader
import sys
import os

# Ajusta las rutas para acceder a los módulos y funciones necesarios
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from datasets.dataset import prepareDatasets  # Ajusta según la estructura de tu proyecto
from training.run_benchmark import custom_collate  # Ajusta según la estructura de tu proyecto

def estimate_memory_usage(dataset_type='FIVES512', dataset_split='train', batch_size=8):
    """
    Estima el uso de memoria de un dataset en la GPU cargando un lote y extrapolando el uso.
    
    Args:
        dataset_type (str): Tipo de dataset ('FIVES512' o 'FIVES1024').
        dataset_split (str): División del dataset ('train', 'val' o 'test').
        batch_size (int): Tamaño del lote para la estimación.

    Returns:
        None, pero imprime la estimación de memoria en la GPU.
    """
    # Selecciona el dispositivo (asegúrate de que sea una GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("Este script necesita una GPU para estimar la memoria correctamente.")
        return

    # Prepara los datasets
    datasets = prepareDatasets()
    if dataset_type not in datasets or dataset_split not in datasets[dataset_type]:
        print(f"Dataset '{dataset_type}' o split '{dataset_split}' no encontrado.")
        return

    # Selecciona el conjunto de datos específico
    dataset = datasets[dataset_type][dataset_split]

    # Crea el DataLoader para el dataset
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    # Carga un solo batch y mide la memoria
    sample_batch = next(iter(data_loader))
    data, label = sample_batch[1].to(device), sample_batch[2].to(device)

    # Calcula el uso de memoria del batch
    batch_memory_usage = torch.cuda.memory_allocated(device)

    # Extrapola al tamaño completo del dataset
    total_memory_estimate = batch_memory_usage * len(data_loader)
    print(f"Uso estimado de memoria para el dataset completo ({dataset_type} - {dataset_split}): {total_memory_estimate / (1024**2):.2f} MB")

if __name__ == "__main__":
    # Argumentos opcionales: tipo de dataset, split y batch_size
    import argparse
    parser = argparse.ArgumentParser(description="Estimación de uso de memoria en GPU para un dataset")
    parser.add_argument('--dataset_type', type=str, default='FIVES512', choices=['FIVES512', 'FIVES1024'],
                        help="Tipo de dataset ('FIVES512' o 'FIVES1024')")
    parser.add_argument('--dataset_split', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Split del dataset ('train', 'val' o 'test')")
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño del batch para la estimación')
    
    args = parser.parse_args()
    estimate_memory_usage(dataset_type=args.dataset_type, dataset_split=args.dataset_split, batch_size=args.batch_size)
