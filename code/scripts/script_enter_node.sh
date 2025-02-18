#!/bin/bash
#SBATCH -J frnet_training        # Nombre del trabajo
#SBATCH -o frnet_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e frnet_error_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita 4 GPUs A100
#SBATCH -c 48                    # 48 núcleos de CPU
#SBATCH --mem=96G                # Memoria total
#SBATCH -t 02:00:00              # Tiempo máximo de ejecución (2 horas)

# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9

# Activar el entorno virtual
source /mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codeBase/FRNet\ para\ FIVES/tfg/bin/activate

# Ejecutar el script de entrenamiento
srun --gres=gpu:a100:1 -c 32 --mem=48G --time=02:00:00 --pty bash
#srun --gres=gpu:a100:4 -c 48 --mem=48G --time=02:00:00 --pty
