import cv2
import os
import numpy as np

def test_load_images(directory):
    failed = []
    for filename in os.listdir(directory):
        # Asegúrate de que el archivo tiene la extensión deseada (ajústala según lo necesites)
        if filename.endswith(".bmp") or filename.endswith(".png") or filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Verificar si la imagen fue cargada correctamente
            if img is None:
                failed.append(filename)
                print(f"Failed to load: {filename} at {filepath}")
            else:
                unique_values = np.unique(img)
                print(f"Loaded successfully: {filename}, Shape: {img.shape}")
                print("Unique values in the image:", unique_values)
                
    return failed

# Especifica la ruta a las imágenes de etiquetas
label_dir = "dataset/FIVES512/val/image"
failed_images = test_load_images(label_dir)
print("Failed images:", failed_images)
