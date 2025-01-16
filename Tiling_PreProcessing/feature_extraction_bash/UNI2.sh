# this script is used to run the feature extraction script for the UNI2 dataset on the cluster.

#!/bin/bash
#SBATCH -c 20
# SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 0-03:00
#SBATCH --mem=50G
#SBATCH --account=yu_ky98
#SBATCH -o debug/job_%A_%a.out
#SBATCH -e debug/job_%A_%a.err
#SBATCH --array=2-35

module load miniconda3/23.1.0
module load gcc/9.2.0
module load cuda/12.1
echo "Modules loaded"
echo "Start"
mkdir -p debug
mkdir -p logs
mkdir -p scripts
source activate /home/idu675/.conda/envs/notebookenv
echo "Conda environment activated"
echo "Activated Python path: $(which python)"

FILE_PATH=$(ls /n/scratch/users/i/idu675/MGB_coords/2_StageInII_MGH_35_Controls_BWHScanner/patch_coord | sed -n "${SLURM_ARRAY_TASK_ID}p")
#FILE_PATH="BD2021_00017901.h5"

export HF_TOKEN=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
echo "Running Python script for file: $FILE_PATH"
/home/idu675/.conda/envs/notebookenv/bin/python /home/idu675/projects/melanoma_Process/feature_UNI.py \
    --coord_path /n/scratch/users/i/idu675/MGB_coords/2_StageInII_MGH_35_Controls_BWHScanner/patch_coord/$FILE_PATH \
    --save_root /n/scratch/users/i/idu675/2_StageInII_MGH_35_Controls_BWHScanner/UNI_features \
    --log_file /n/scratch/users/i/idu675/2_StageInII_MGH_35_Controls_BWHScanner/UNI_features/extraction_${SLURM_ARRAY_TASK_ID}.log \
    > logs/compute_coords_${SLURM_ARRAY_TASK_ID}.log 2>&1
echo "Python script finished for file: $FILE_PATH"
