import cv2
from PIL import Image
import os
import numpy as np

def check_image_accessibility(filepath):
    """
    Verifica si la imagen puede abrirse y carga información usando PIL y luego intenta con cv2.
    """
    try:
        # Abre la imagen con PIL y obtiene información básica
        with Image.open(filepath) as img:
            img_format = img.format
            img_mode = img.mode
            img_size = img.size
            print(f"Información PIL - Formato: {img_format}, Modo: {img_mode}, Tamaño: {img_size}")
            img.verify()  # Verifica si la imagen está completa y no está corrupta
            print("La imagen se verificó correctamente con PIL.")
    except Exception as e:
        print(f"Error al abrir la imagen con PIL: {e}")
        return False

    # Verificar permisos de lectura
    if not os.access(filepath, os.R_OK):
        print(f"Error: No se tienen permisos de lectura para la imagen en {filepath}")
        return False

    # Intentar cargar la imagen con cv2
    img_cv2 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_cv2 is None:
        print(f"Error: cv2.imread() no pudo cargar la imagen en {filepath}")
        return False

    print("La imagen se cargó exitosamente con cv2.")
    print("Dimensiones de la imagen con cv2:", img_cv2.shape)
    return True

def test_load_images(directory):
    failed = []
    for filename in os.listdir(directory):
        # Incluye más formatos de imagen por si acaso
        if filename.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            filepath = os.path.join(directory, filename)
            print(filepath)
            print(f"\nVerificando imagen: {filename}")
            if not check_image_accessibility(filepath):
                failed.append(filename)
    return failed

# Especifica la ruta a las imágenes de etiquetas
label_dir = "dataset/FIVES512/val/image"
failed_images = test_load_images(label_dir)
print("\nImágenes que fallaron:", failed_images)
