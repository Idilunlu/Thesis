
import os
import enum
import torch
import torch.nn as nn
from torchvision import transforms 
from typing import Optional
from huggingface_hub import hf_hub_download, login
from transforms import get_transforms
#from uni import get_encoder
import h5py
import timm
from timm.models.vision_transformer import VisionTransformer
from torch.hub import load_state_dict_from_url
from transformers import CLIPModel, SwinModel, ViTModel
import openslide
import argparse
from tqdm import tqdm
import logging

hf_token = "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"

def get_uni(hf_token: Optional[str] = None):
    """
    =============== UNI ===============
    https://huggingface.co/MahmoodLab/UNI
    Warning: this model requires an access request to the model owner.
    =============== UNI ===============
    """
    if hf_token:
        login(token=hf_token)

    model_dir = os.path.expanduser("~/.models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = hf_hub_download(
        "MahmoodLab/UNI",
        filename="pytorch_model.bin",
        cache_dir=model_dir,
        force_download=False,
    )
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    return model
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--coord_path', type=str, required=True, help="Path to the .h5 file")
    parser.add_argument('--save_root', type=str, required=True, help="Root directory to save output")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--log_file', type=str, default="logfile.log")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    # Set up logging
    log_file_path = args.log_file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,  # Log to file specified by user
        filemode='w'  # Overwrite the log file each run
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    device = "cuda" if args.gpu else "cpu"
    logging.info(f"Using device: {device}")
    logging.info(f"Processing file: {args.coord_path}")
    logging.info(f"Saving output to: {args.save_root}")

    # model, transform = get_encoder(enc_name='uni', device=device)
    model = get_uni(hf_token)
    transform = get_transforms(False)
    model = model.eval()

    file = args.coord_path
    file_name = os.path.basename(file).split(".h5")[0]

    try:
        output_tensor = None
        with h5py.File(file, 'r') as f:
            wsi_file = f['coords'].attrs['path']
            wsi = openslide.OpenSlide(wsi_file)
            data = f['coords'][()]
            c = data.shape[0]
            logging.info(f"Extracting {c} patches from WSI file: {wsi_file}")

            for idx in range(c):
                image = wsi.read_region(data[idx], 0, (args.patch_size, args.patch_size)).convert('RGB')
                image = transform(image).unsqueeze(dim=0).to(device)
                with torch.no_grad():
                    patch_feature_emb = model(image)  # Extracted features (torch.Tensor) with shape [1,1024]
                if idx == 0:
                    output_tensor = patch_feature_emb
                else:
                    output_tensor = torch.cat((output_tensor, patch_feature_emb), dim=0)

            # Save file
            os.makedirs(args.save_root, exist_ok=True)
            output_file_path = os.path.join(args.save_root, f"{file_name}.pt")
            torch.save(output_tensor, output_file_path)
            logging.info(f"Saved features to {output_file_path}")

    except Exception as e:
        logging.error(f"Failed to process file {file}: {str(e)}")
