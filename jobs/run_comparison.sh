#!/bin/bash
#SBATCH --job-name=phenology_comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/phenology_comp_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/phenology_comp_%j.err
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

# Print some information about the job
echo "Running on host $(hostname)"
echo "Starting time: $(date)"
echo "Directory: $(pwd)"

# Load required modules
module purge
module load python/3.9.12
module load gdal/3.3.3

# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Create output directory
OUTPUT_DIR="/lustre/fswork/projects/rech/ego/uyr48jk/InferencePhenology/results/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run the comparison script
python $WORK/InferencePhenology/src/run_comparison.py \
    --output-dir $OUTPUT_DIR \
    --window-size 5120 \
    --max-workers $SLURM_CPUS_PER_TASK

echo "Job finished: $(date)"