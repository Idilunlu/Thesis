#!/bin/bash
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -t 0-6:00
#SBATCH --mem=60G
#SBATCH --account=yu_ky98
#SBATCH -o debug/job_%A_%a.out
#SBATCH -e debug/job_%A_%a.err

# Setting model and data directories for Dreambooth-LoRA training

# export INSTANCE_DIR="/n/data2/hms/dbmi/kyu/lab/che099/data/frozen_patches_20x_mutated_pngs/00620C3D-01C2-4487-B556-44697751DCCE"
#dir_path="/n/data2/hms/dbmi/kyu/lab/che099/data/frozen_patches_20x_wildtype_pngs"
#INSTANCE_FILES=$(ls "$dir_path" | head -50 | awk -v prefix="$dir_path/" '{print prefix $0}' ORS=':')
#INSTANCE_FILES=${INSTANCE_FILES%:}
#export INSTANCE_DIR=$INSTANCE_FILES

#export CLASS_DATA_DIR="n/data2/hms/dbmi/kyu/lab/che099/data/frozen_patches_20x_mutated_pngs/00e68225-1fd3-48e2-92f0-9ad0c3b8302c"
#class_path="/n/data2/hms/dbmi/kyu/lab/che099/data/frozen_patches_20x_wildtype_pngs"
#CLASS_DATA_DIR=$(ls "$class_path" | head -100 | awk -v prefix="$class_path/" '{print prefix $0}' ORS=':')
#CLASS_DATA_DIR=${CLASS_DATA_DIR%:}
#export CLASS_DATA_DIR=$CLASS_DATA_DIR

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/n/data2/hms/dbmi/kyu/lab/che099/data/instance_files"
export CLASS_DATA_DIR="/n/data2/hms/dbmi/kyu/lab/che099/data/class_files"
export OUTPUT_DIR="/home/idu675/projects/Thesis/catDreambooth/outputs_extensive"
export LOGGING_DIR="/home/idu675/projects/Thesis/catDreambooth/logs_extensive"

# Launch the Dreambooth training script with specified parameters
accelerate launch catDreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="Generate a high-resolution frozen brain histopathology image of a 20-year-old woman diagnosed with IDH-mutant diffuse astrocytoma (WHO grade II). Use 20x magnification with H&E staining, and create an image that has high contrast, and natural color balance. Include moderately increased cellularity compared to normal brain tissue, diffusely infiltrating astrocytic tumor cells with mild to moderate nuclear atypia, a fibrillary background with fine processes of tumor cells, and microcystic changes characteristic of low-grade astrocytomas. Ensure the absence of significant mitotic activity, microvascular proliferation, or necrosis. Include a small area of adjacent normal brain tissue for comparison. The image should clearly show nuclear details and cellular processes, representing the typical appearance of a low-grade, IDH-mutant astrocytoma in a young adult patient." \
  --class_data_dir=$CLASS_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=8 \
  --num_train_epochs=200 \
  --learning_rate=1e-4 \
  --with_prior_preservation \
  --gradient_accumulation_steps=4 \
  --mixed_precision="fp16" \
  --logging_dir=$LOGGING_DIR \
  --validation_epochs=4 \
  --validation_prompt="Generate a high-resolution frozen brain histopathology image of a 20-year-old woman diagnosed with IDH-mutant diffuse astrocytoma (WHO grade II). Use 20x magnification with H&E staining, and create an image that has high contrast, and natural color balance. Include moderately increased cellularity compared to normal brain tissue, diffusely infiltrating astrocytic tumor cells with mild to moderate nuclear atypia, a fibrillary background with fine processes of tumor cells, and microcystic changes characteristic of low-grade astrocytomas. Ensure the absence of significant mitotic activity, microvascular proliferation, or necrosis. Include a small area of adjacent normal brain tissue for comparison. The image should clearly show nuclear details and cellular processes, representing the typical appearance of a low-grade, IDH-mutant astrocytoma in a young adult patient." \
  --dataloader_num_workers=8 \
  --seed=42 \
  --class_prompt="Generate a high resolution frozen brain histopathology slide of a young patient diagnosed with IDH mutation and low grade glioma with 20x magnification" \
  --report_to="wandb"
