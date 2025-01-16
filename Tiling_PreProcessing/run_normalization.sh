# batch script to run the normalization.py script on the O2 cluster 

#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2:00:00                         # Runtime in D-HH:MM format
#SBATCH -p short                             # Partition to run in
#SBATCH --mem=4G                          # Memory total in MiB (for all cores)
#SBATCH -o ./image_normalization/logs/normalize_patches_PM_%A_%a.log                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./image_normalization/logs/normalize_patches_PM_%A_%a.log                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=yu_ky98
#SBATCH --array=0-6          # Run array for indexes 1-4, effectively running the command 4x (but with different input files)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

source activate /home/idu675/.conda/envs/notebookenv

### 
# csv file containing the list of tiles to normalize
#CSV_FILE=data_tile_csv/TVGH_FS_GBM_PCNSL_top1000_labels.csv
CSV_FILE=/n/scratch/users/i/idu675/3_LSP_PCA_project_BWHScaner.csv
###
# output path for normalized tiles
OUTPATH=/n/scratch/users/i/idu675/normalized_3_LSP_PCA_project_BWHScaner

###
# (for parallelization using O2 job arrays)
#  N is the number of tiles to normalize per job
N=10000


ISTART=$((SLURM_ARRAY_TASK_ID * N))
IEND=$(((SLURM_ARRAY_TASK_ID + 1) * N))
echo "ISTART: ${ISTART}"
echo "IEND: ${IEND}"

/home/idu675/.conda/envs/notebookenv/bin/python /home/idu675/projects/melanoma_Process/normalization.py \
 --csv_file ${CSV_FILE} --outpath ${OUTPATH} --i_start ${ISTART} --i_end ${IEND}

