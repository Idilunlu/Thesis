# This script takes a single H5 file path instead of a folder path to iterate over
# Since sbatch array iterates over the H5 files, this version of feature extraction is more convenient for parallel processing

#! /home/idu675/.conda/envs/notebookenv/bin/python

import os
import torch
import torch.nn as nn
import pickle
from torchvision import transforms
#from models.ctran import ctranspath
#from models.ctrans import ctranspath
import timm
import h5py
import openslide
from glob import glob
import argparse
from tqdm import tqdm
import logging

# CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"
CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"




def get_model(model_pth: str):
    import oldtimm
    from oldtimm.models.layers.helpers import to_2tuple
    import torch.nn as nn

    class ConvStem(nn.Module):

        def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
        ):
            super().__init__()

            assert patch_size == 4
            assert embed_dim % 8 == 0

            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = (
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.flatten = flatten

            stem = []
            input_dim, output_dim = 3, embed_dim // 8
            for l in range(2):
                stem.append(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                stem.append(nn.BatchNorm2d(output_dim))
                stem.append(nn.ReLU(inplace=True))
                input_dim = output_dim
                output_dim *= 2
            stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
            self.proj = nn.Sequential(*stem)

            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        def forward(self, x):
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
            return x

    model = oldtimm.create_model(
        "swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False
    )
    model.head = nn.Identity()
    td = torch.load(model_pth)
    model.load_state_dict(td["model"], strict=True)
    model = model.eval()
    return model


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cancer', default="01_BRCA", type=str)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--coord_path', type=str, default=None)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--log_file', type=str, default="logfile.log")
    # parser.add_argument('--slide', type=str, default="PM")
    args = parser.parse_args()
    return args

def img_transform(patch_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    return trnsfrms_val

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

   # device = "cuda" if args.gpu else "cpu"
   # logging.info(f"Using device: {device}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # cancer_type = args.cancer # e.g., '04_LUAD'
    #logging.info(f"Processing cancer type: {cancer_type}")
    logging.info(f"Saving output to: {args.save_root}")

    # model = ctranspath()
    # model.head = nn.Identity()
    # td = torch.load(r'./model_weight/CHIEF_CTransPath.pth')
    # model.load_state_dict(td['model'], strict=True)
    
    model = get_model(CHIEF_PTH)
    model.eval().to(device)

    # Obtain coordinates of valid patches
    logging.info(f"Searching for h5 files in path: {args.coord_path}")    
    # h5file = glob(os.path.join(args.coord_path,"*.h5"))

    file = args.coord_path
    file_name = os.path.basename(file).split(".h5")[0]
    logging.info(f"Processing file: {file}")

    
    # logging.info(f"coord_path set to: {args.coord_path}")
    logging.info(f"save_root set to: {args.save_root}")

    # Image settings
    patch_size = args.patch_size
    trnsfrms_val = img_transform(patch_size)

    try:
        output_tensor = None
        with h5py.File(file, 'r') as f:
            wsi_file = f['coords'].attrs['path']
            wsi = openslide.OpenSlide(wsi_file)
            data = f['coords'][()]
            c = data.shape[0]
            logging.info(f"Extracting {c} patches from WSI file: {wsi_file}")

            for idx in range(c):
                image = wsi.read_region(data[idx], 0, (patch_size, patch_size)).convert('RGB')
                image = trnsfrms_val(image).unsqueeze(dim=0).to(device)
                with torch.no_grad():
                    patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
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
