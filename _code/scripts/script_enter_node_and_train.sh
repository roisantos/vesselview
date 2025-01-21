#!/bin/bash
srun --gres=gpu:a100:2 -c 64 --mem=64G --time=04:00:00 --pty bash

cd /mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga
# Activar el entorno virtual
source /mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/tfg/bin/activate

# Ejecutar el script de entrenamiento
python3 code/training/run_benchmark.py
#srun --gres=gpu:a100:4 -c 48 --mem=48G --time=02:00:00 --pty
