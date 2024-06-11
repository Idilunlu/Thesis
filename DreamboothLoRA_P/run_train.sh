#!/bin/bash

# "stabilityai/stable-diffusion-2-1-base" 768
# "runwayml/stable-diffusion-v1-5" 512

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="LU.ruAD_black_formalin_origin/*.jpg"
export OUTPUT_DIR="LUAD_black_formalin"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="LUAD black formalin" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=2000 \
  --validation_prompt="a photo of LUAD black formalin" \
  --validation_epochs=1000 \
  --seed="0"
  