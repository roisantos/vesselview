#!/bin/bash
#SBATCH -J Gamma1.5       # Nombre del trabajo
#SBATCH -o RoiNet9_FIVESc_FocalTverskyGamma1_gein_output_%j.log   # Archivo para la salida estándar (%j expande al JobID)
#SBATCH -e RoiNet9_FIVESc_FocalTverskyGamma1_gein_%j.log    # Archivo para la salida de errores
#SBATCH --gres=gpu:a100:1        # Solicita GPU A100
#SBATCH -c 32                    # 32 núcleos de CPU
#SBATCH --mem=32G                # Memoria total
#SBATCH -p medium
#SBATCH -t 3-00:00:00              # Tiempo máximo de ejecución

# Cargar módulos necesarios
module load cesga/2020
module load python/3.9.9


cd /mnt/netapp2/Store_uni/home/usc/ec/rsm/fivesegmentor
source ../vroi/bin/activate


# User-defined configuration variables
MODEL="RoiNet9"
DATASET="FIVESc"
CONFIG="code/config/config.json"
EPOCHS=300
EARLY_STOP=100
BATCH_SIZE=1
NUM_WORKERS=32
LR=1e-4
WEIGHT_DECAY=0.001
LOSS="FocalTversky"
LOGGING="True"
OUTPUT_PREFIX="RoiNet9_FIVESc_FocalTverskyGamma1_gein_"
THRESH_VALUE=100

# Augmentation variables
AUGMENT_GEOMETRIC="True" 
AUGMENT_ELASTIC="True"
AUGMENT_INTENSITY="True"
AUGMENT_GAMMA="False"
AUGMENT_NOISE="True"

# Restormer
RESTORMER="False"



# Launch the training script with the specified parameters
python3 code/training/run_benchmark.py \
  -model "$MODEL" \
  -dataset "$DATASET" \
  --config "$CONFIG" \
  --epochs "$EPOCHS" \
  --early_stopping "$EARLY_STOP" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --loss "$LOSS" \
  --logging "$LOGGING" \
  --output_prefix "$OUTPUT_PREFIX" \
  --thresh_value "$THRESH_VALUE" \
  --augment_geometric "$AUGMENT_GEOMETRIC" \
  --augment_elastic "$AUGMENT_ELASTIC" \
  --augment_intensity "$AUGMENT_INTENSITY" \
  --augment_gamma "$AUGMENT_GAMMA" \
  --augment_noise "$AUGMENT_NOISE" \
  --restormer "$RESTORMER"
