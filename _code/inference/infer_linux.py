import cv2
import torch
import os
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))
from models.frnet import FRNet
from models.common import *
from torchvision.utils import save_image

def run_inference_on_directory(input_dir, output_dir, model_path):
    # Verificar si el archivo de modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Crear una instancia del modelo
    model = FRNet(ch_in=1, ch_out=1, ls_mid_ch=[32]*6, out_k_size=11, k_size=3,
                  cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock)

    # Cargar los pesos entrenados del modelo
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Poner el modelo en modo de inferencia

    # Comprobar si hay GPU disponible, si no, usar CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Asegurarse de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Procesar cada archivo en el directorio de entrada
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Cargar la imagen
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Advertencia: Saltando archivo debido a fallo de carga: {file_path}")
            continue

        # Normalizar la imagen
        image = image.astype("float32") / 255

        # Aplicar padding para que las dimensiones sean múltiplos de 32
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)

        # Convertir la imagen a un tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Agregar dimensiones de batch y canal
        image_tensor = image_tensor.to(device)

        # Realizar la inferencia
        with torch.no_grad():
            output = model(image_tensor)

        # Guardar el tensor y la imagen
        torch.save(output.cpu(), os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output_tensor.pth"))
        save_image(output, os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output_image.png"))

# Directorio que contiene las imágenes (ajusta la ruta a tu entorno de trabajo)
input_dir = r"dataset/FIVES512/test/scaffoldsCENTER1024masked/masked"
# Directorio para guardar los resultados
output_dir = os.path.join('inference_results_test', '2024-16-10_test2')
# Ruta al modelo entrenado (ajusta la ruta a tu entorno de trabajo)
model_path = 'result/FRNet-base/FIVES512/model_best.pth'

run_inference_on_directory(input_dir, output_dir, model_path)
