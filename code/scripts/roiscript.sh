#!/bin/bash
#SBATCH -J RoiNet        # Nombre del trabajo
#SBATCH -o RoiNet_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e RoiNet_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita 4 GPU A100
#SBATCH -c 32                    # 32 núcleos de CPU
#SBATCH --mem=16G                # Memoria total
#SBATCH -p medium
#SBATCH -t 3-00:00:00              # Tiempo máximo de ejecución (2 horas)
# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9


cd /home/usc/ec/rsm/fivesegmentor/
source ../vroi/bin/activate
# Ejecutar el script de entrenamiento
srun python ./code/training/run_benchmark.py -model "RoiNet"
