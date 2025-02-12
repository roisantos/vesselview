#!/bin/bash
#SBATCH -J Rs7x10        # Nombre del trabajo
#SBATCH -o Rs7x10_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e Rs7x10_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita 4 GPU A100
#SBATCH -c 32                    # 32 núcleos de CPU
#SBATCH --mem=32G                # Memoria total
#SBATCH -p medium
#SBATCH -t 3-00:00:00              # Tiempo máximo de ejecución (2 horas)
# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9


cd /home/usc/ec/rsm/fivesegmentor/
source ../vroi/bin/activate
# Ejecutar el script de entrenamiento
srun python ./code/training/run_benchmark.py -model "RoiNet_soft_dice_cldice"
