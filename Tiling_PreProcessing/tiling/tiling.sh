#!/bin/bash

# Define your main directories
MAIN_DIR="/n/data2/hms/dbmi/kyu/lab/bok448/datasets/SKCM-Stage-1-2/downloads"
CSV_DIR="/n/scratch/users/i/idu675"
SAVE_ROOT="/n/scratch/users/i/idu675/MGB_coords"
LOG_DIR="/n/scratch/users/i/idu675/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# List of directories to process
DIRECTORIES=("1_StageInII_Melanoma_BWHScaner" "2_StageInII_MGH_35_Controls_BWHScanner" "3_LSP_PCA_project_BWHScaner")

# Iterate over each directory
for DIR in "${DIRECTORIES[@]}"; do
    # Define paths
    CSV_PATH="${CSV_DIR}/${DIR}.csv"
    SAVE_PATH="${SAVE_ROOT}/${DIR}"

    # Ensure save directory exists
    mkdir -p "$SAVE_PATH"

    # Define log file
    LOG_FILE="${LOG_DIR}/compute_coords-${DIR}.log"

    # Construct and run the command
    python3 tiling.py \
        --save_root "$SAVE_PATH" \
        --csv_path "$CSV_PATH" \
        --target_mpp 1.0 \
        --patch_size 224 \
        --downsample 16 \
        --threshold 0.15 \
        --num_worker 20 \
        --n_part 1 \
        --part 0 > "$LOG_FILE" 2>&1
done