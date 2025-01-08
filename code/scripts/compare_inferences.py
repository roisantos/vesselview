import time
import torch
import cv2

import sys
import os

# Ajusta las rutas para acceder a los módulos y funciones necesarios
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)


from torchvision.utils import save_image
from models.frnet import FRNet
from models.common import ResidualBlock

def load_model(model_path):
    """
    Cargar el modelo FRNet con pesos desde un archivo .pth.
    """
    model = FRNet(ch_in=1, ch_out=1, ls_mid_ch=[32] * 6, out_k_size=11, k_size=3,
                  cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def run_inference(model, input_dir, output_dir, device):
    """
    Realiza la inferencia sobre todas las imágenes de `input_dir` y guarda los resultados en `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    times = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Advertencia: Saltando archivo debido a fallo de carga: {file_path}")
            continue
        
        image = image.astype("float32") / 255.0
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convertir a milisegundos

        save_image(output.cpu(), os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output_image.png"))

    avg_time = sum(times) / len(times)
    print(f"Tiempo promedio de inferencia: {avg_time:.2f} ms")
    return avg_time

def compare_models(model1_path, model2_path, input_dir_512, input_dir_1024):
    """
    Compara los tiempos de inferencia entre dos modelos, cada uno con su conjunto de imágenes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "infer_compare"
    os.makedirs(output_dir, exist_ok=True)

    # Inferencia para el modelo de 512
    print("Comparando modelo de 512...")
    model1 = load_model(model1_path)
    model1_output_dir = os.path.join(output_dir, "model512_outputs")
    avg_time_model1 = run_inference(model1, input_dir_512, model1_output_dir, device)

    # Inferencia para el modelo de 1024
    print("Comparando modelo de 1024...")
    model2 = load_model(model2_path)
    model2_output_dir = os.path.join(output_dir, "model1024_outputs")
    avg_time_model2 = run_inference(model2, input_dir_1024, model2_output_dir, device)

    print("\nResultados de comparación de tiempo de inferencia:")
    print(f"Tiempo promedio Modelo 512: {avg_time_model1:.2f} ms")
    print(f"Tiempo promedio Modelo 1024: {avg_time_model2:.2f} ms")

# Rutas a los modelos y a los directorios de imágenes de entrada para cada resolución
model512_path = '/mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/run_benchmark_runs/run512_result_2024-11-09_12-30-18/FRNet/model_best.pth'  # Cambia esta ruta al modelo FRNet de 512
model1024_path = '/mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/run_benchmark_runs/run1024_result_2024-11-08_19-43-11/FRNet-base/model_best.pth'  # Cambia esta ruta al modelo FRNet de 1024
input_dir_512 = '/mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/dataset/FIVES512/test/image'
input_dir_1024 = '/mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/dataset/FIVES1024/test/image'

compare_models(model512_path, model1024_path, input_dir_512, input_dir_1024)
