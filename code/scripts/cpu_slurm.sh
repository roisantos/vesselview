#!/bin/bash
#SBATCH -J frnet_training        # Nombre del trabajo
#SBATCH -o frnet_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e frnet_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:4        # Solicita 4 GPUs A100
#SBATCH -c 48                    # 48 núcleos de CPU
#SBATCH --mem=48G                # Memoria total
#SBATCH -t 04:00:00              # Tiempo máximo de ejecución (2 horas)
# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9


Ejecutar el script de entrenamiento
srun python3 ../training/run_benchmark.py
#srun --gres=gpu:a100:4 -c 48 --mem=48G --time=02:00:00 --pty
