#!/bin/bash



export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

export OUTPUT_DIR=LUSC_Black_individual_generator_strength06/TCGA-6A-AB49-01Z-00-DX1.FDF2EED7-57A3-4019-A382-21DED11780F6
#STEP=(2500 3000 3500)

#for step in ${STEP[@]};
#do
    #export LORA_NAME="kaneyxx/black_LUAD_2x_"$step"_1e-4"
    #export LORA_NAME="LUSC_black_formalin/pytorch_lora_weights.bin"
    export LORA_NAME="/Users/idilunlu/Downloads/pytorch_lora_weights.bin"
    # I need to upload weights to my scratch account and put its address here
    #echo "Current repo:" $LORA_NAME
    python image_generation.py \
      --pretrained_lora=$LORA_NAME  \
      --inference_prompt="a photo of LUAD black formalin" \
      --output_dir=$OUTPUT_DIR \
      --sample_batch_size=5 \
      --sample_batch_count=400 \
      --inference_steps=25
    #done
