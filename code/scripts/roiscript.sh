#!/bin/bash

# Define the model name (Change this variable to update everything)
MODEL_NAME="RoiNetx1.5"

#SBATCH -J ${MODEL_NAME}        # Job name dynamically set
#SBATCH -o ${MODEL_NAME}_output_%j.log   # Output log file
#SBATCH -e ${MODEL_NAME}_error_%j.log    # Error log file
#SBATCH --gres=gpu:a100:1        # Request 1 GPU A100
#SBATCH -c 32                    # 32 CPU cores
#SBATCH --mem=16G                # Total memory
#SBATCH -p medium
#SBATCH -t 3-00:00:00            # Max execution time (3 days)

# Load necessary modules
module load cesga/2020
module load python/3.9.9

cd /home/usc/ec/rsm/fivesegmentor/
source ../vroi/bin/activate

# Execute training script with the model
srun python ./code/training/run_benchmark.py -model "${MODEL_NAME}"
