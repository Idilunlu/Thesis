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
export OUTPUT_DIR="/home/idu675/projects/Thesis/catDreambooth/outputs"
export LOGGING_DIR="/home/idu675/projects/Thesis/catDreambooth/logs"

# Launch the Dreambooth training script with specified parameters
accelerate launch Dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="High-resolution histopathology image of IDH-wildtype glioma. (H&E
stain:1.2), 20x magnification. Display (increased cellularity:1.3) with (more pleomorphic
tumor cells:1.4). Show (moderate to marked nuclear atypia:1.3), (mitotic figures:1.2), and
(microvascular proliferation:1.3). Include (areas of necrosis:1.2) if present. (Clear nuclear
details:1.3), (hyperchromatic nuclei:1.2), (nuclear pleomorphism:1.3). (Infiltrative growth
pattern:1.2." \
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
  --validation_prompt="High-resolution histopathology image of IDH-wildtype glioma. (H&E
stain:1.2), 20x magnification. Display (increased cellularity:1.3) with (more pleomorphic
tumor cells:1.4). Show (moderate to marked nuclear atypia:1.3), (mitotic figures:1.2), and
(microvascular proliferation:1.3). Include (areas of necrosis:1.2) if present. (Clear nuclear
details:1.3), (hyperchromatic nuclei:1.2), (nuclear pleomorphism:1.3). (Infiltrative growth
pattern:1.2" \
  --dataloader_num_workers=8 \
  --seed=42 \
  --class_prompt="Generate a high resolution frozen brain histopathology slide of a young patient diagnosed with IDH mutation and low grade glioma with 20x magnification" \
  --report_to="wandb"
