import os
import argparse
from pathlib import Path
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler
)

from PIL import Image

import glob
import random

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_lora",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained lora from huggingface.co/models.",
    )
    parser.add_argument(
        "--inference_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--sample_batch_size", 
        type=int, 
        default=4, 
        help="Batch size for sampling images."
    )
    parser.add_argument(
        "--sample_batch_count", 
        type=int, 
        default=1, 
        help="Batch count for sampling images."
    )
    parser.add_argument(
        "--inference_steps", 
        type=int, 
        default=25, 
        help="Steps for inference"
    )
    parser.add_argument(
        "--img_source", 
        type=str, 
        default=None, 
        help="img_source input"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args


def main(args):
    accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=None,
            log_with=None,
            logging_dir=None,
        )

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                revision=None, 
                torch_dtype=torch.float32
            )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    print("Pipeline has been established. ")
    
    # load attention processors
    pipeline.unet.load_attn_procs(args.pretrained_lora)
    print("LoRA weight loaded.")
    
    # run inference and save
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    #generator = torch.Generator(device="cpu").manual_seed(0)
    lora_name = args.pretrained_lora.split("/")[-1]
    prompt = args.sample_batch_size * [args.inference_prompt]
    generated_image_dir = Path(args.output_dir) 
    #generated_image_dir.mkdir()
    print("inference...")

    # 指定要遍历的文件夹路径
    #the_folder_path = 'LUSC_Black_individual_origin/TCGA-6A-AB49-01Z-00-DX1.FDF2EED7-57A3-4019-A382-21DED11780F6'
    the_folder_path = 'LU.ruAD_black_formalin_origin'
    # 使用 glob 模块匹配所有以 .jpg 结尾的文件
    jpg_files = glob.glob(os.path.join(the_folder_path, '*.jpg'))
    # print(jpg_files)
    for i in range(args.sample_batch_count):
        random_number = random.randint(0, 9)
        image_path = jpg_files[random_number]  # 使用图像文件的完整路径
        #image_path = "LUAD_black_formalin_origin/TCGA-95-7562-01A-01-TS1.1ba56ba5-ecd4-4eb1-ac49-3d8108114b72_13_7.jpg"  # 验证给定的图片路径
        image = Image.open(image_path)
        images = pipeline(prompt, image, num_inference_steps=args.inference_steps, strength=0.6, generator=generator).images
        #images = pipeline(prompt, num_inference_steps=args.inference_steps, generator=generator).images
        for j, image in enumerate(images):
            out_path = generated_image_dir / f'image_{i}_{j}.png'
            image.save(out_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)