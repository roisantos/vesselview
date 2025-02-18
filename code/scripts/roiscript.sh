#!/bin/bash
#SBATCH -J restoa        # Nombre del trabajo
#SBATCH -o restoa_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e restoa_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita 4 GPU A100
#SBATCH -c 32                    # 32 núcleos de CPU
#SBATCH --mem=32G                # Memoria total
#SBATCH -p medium
#SBATCH -t 3-00:00:00              # Tiempo máximo de ejecución (2 horas)
# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9

cd /mnt/netapp2/Store_uni/home/usc/ec/rsm/restoa/
source ../vroi/bin/activate
# Ejecutar el script de entrenamiento
srun python ./code/training/run_benchmark.py -model "RoiNet_Dice"
