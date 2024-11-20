#!/bin/bash
#SBATCH --job-name=prep_inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.err

echo '### Running $SLUM_JOB_NAME ###'
set -x

source $HOME/.bashrc
module load gdal
module load pytorch-gpu/py3/2.2.0

python $WORK/InferencePhenology/src_inference/prepare_inference.py \
    --mosaic-dir /lustre/fsn1/projects/rech/ego/uyr48jk/all_year_france/gpfsscratch/rech/ego/uof45xi/data/all_year/france \
    --dem-dir /lustre/fsn1/projects/rech/ego/uyr48jk/all_year_france/gpfsscratch/rech/ego/uof45xi/data/all_year/france/dem \
    --output-dir /lustre/fswork/projects/rech/ego/uyr48jk/InferencePhenology \
    --model-path /lustre/fswork/projects/rech/ego/uyr48jk/InferencePhenology/model/best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl \
    --years 2021 2022 \
    --max-concurrent-jobs 100 \
    --num-harmonics 2 \
    --max-iter 1 \
    --window-size 1024 \
    --workers-per-tile 4