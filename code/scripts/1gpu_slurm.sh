#!/bin/bash
#SBATCH -J 4layer64ch_training        # Nombre del trabajo
#SBATCH -o 4layer64ch_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e 4layer64ch_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita 1 GPU A100
#SBATCH -c 32                    # 32 núcleos de CPU
#SBATCH --mem=48G                # Memoria total
#SBATCH -t 47:59:00              # Tiempo máximo de ejecución (2 horas)
# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9


cd /home/usc/ec/rsm/tfg_codebase_cesga/
source venv/bin/activate
# Ejecutar el script de entrenamiento
srun python3 ./code/training/run_benchmark.py
#srun --gres=gpu:a100:1 -c 32 --mem=48G --time=47:59:00 --pty
