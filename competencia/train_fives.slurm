#!/bin/bash
#SBATCH -J WNet        # Job name: Retina_FIVES_<MODEL_NAME>
#SBATCH -o fives_WNet_output_%j.log   # Output file: fives_<MODEL_NAME>_output_<JOB_ID>.log
#SBATCH -e fives_WNet_error_%j.log    # Error file:  fives_<MODEL_NAME>_error_<JOB_ID>.log
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH -c 32                   # Request 32 CPU cores
#SBATCH --mem=32G               # Request 32GB of RAM
#SBATCH -p medium               # Partition (queue) - CHANGE THIS IF NEEDED
#SBATCH -t 3-00:00:00           # Time limit (3 days - adjust as needed)

# --- Environment Setup (CRITICAL: Adapt these paths!) ---

# 1. Load necessary modules (adjust for your cluster)
module load cesga/2020
module load python/3.9.9

# 2. Activate your virtual environment (adjust the path)
source ../../vcompetencia/bin/activate

# 3. Change directory to your project directory (adjust the path)
cd /home/usc/ec/rsm/fivesegmentor/competencia

# --- Run the Training Script ---

# Use train_baseline.py for FIVES, with --config and --model
# Available models (from networks/__init__.py): WNet, Unet, FR_UNet, SA_UNet, MANet
srun python train_baseline.py --config config/fives.yaml

# --- Notes ---
# - The $1 takes the first argument passed to the slurm script (the model name).
# - Example usage:  sbatch train_fives.slurm WNet
# - This assumes fives.yaml exists and has appropriate settings (except model type).
# - Trained models (checkpoints) will be saved in:
#   /mnt/qb/berens/users/jfadugba97/RetinaSegmentation/model_results/<MODEL_TYPE>/<DATASET_TYPE>/<LOSS_TYPE>/
#   For example: /mnt/qb/berens/users/jfadugba97/RetinaSegmentation/model_results/WNet/FIVES/DiceLoss/checkpoint-epochXXX.pth
