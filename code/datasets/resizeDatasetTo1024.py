import os
from PIL import Image

def resize_images_in_directory(root_dir, target_size=(1024, 1024)):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:  # Corrected variable name from 'file_metaphor' to 'file_path'
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img.save(file_path)  # Corrected variable name from 'file_expression' to 'file_path'
                print(f"Resized {file} to {target_size} and saved at {file_path}")  # Corrected variables in the print statement

if __name__ == "__main__":
    dataset_directory = 'dataset/FIVES1024'  # Adjust the path to your dataset directory
    resize_images_in_directory(dataset_directory)
