#!/bin/bash
#SBATCH --job-name=phenology
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4            # 4 CPU cores per tile as requested
#SBATCH --partition=cpu_p1           # Using cpu_p1 for longer jobs
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00             # 20 hours max
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-280%100            # Will be adjusted based on number of tiles

echo '### Running Phenology Inference - Task ${SLURM_ARRAY_TASK_ID} ###'
set -x

source $HOME/.bashrc
module load gdal
module load pytorch-gpu/py3/2.2.0

# Get inference directory from prepare_inference output
INFERENCE_DIR="/lustre/fswork/projects/rech/ego/uyr48jk/InferencePhenology/phenology_inference"

# Get tile index from SLURM array task ID
TILE_IDX=${SLURM_ARRAY_TASK_ID}

# Run inference for the assigned tile
python $WORK/InferencePhenology/src_inference/run_inference.py \
    --config-dir ${INFERENCE_DIR}/configs \
    --tile-idx ${TILE_IDX}