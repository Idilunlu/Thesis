#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding=utf-8
#
# This file is adapted from https://github.com/huggingface/diffusers/blob/febaf863026bd014b7a14349336544fc109d0f57/examples/dreambooth/train_dreambooth_lora.py
# The original license is as below:
#
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


# In[7]:


import argparse
import hashlib
import logging
import math
import os
import warnings
print("Importing Path")
from pathlib import Path
print("Path imports successful.")
from typing import Optional
from transformers import CLIPTextModel
import torch.nn as nn


# In[8]:


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset


# In[9]:


import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
# from pathlib import Path
from accelerate.utils import is_wandb_available
from torch.utils.data import DataLoader 


# In[10]:


from glob import glob


# In[11]:


import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


# In[12]:


from torch.utils.data import Dataset
import pandas as pd
from typing import List, Optional, Callable
import h5py
# from utils.helpers import get_transforms, seed_everything
# from utils.dataset import load_success_ids
from random import sample
from tqdm import tqdm
from PIL import Image
import sqlite3


# In[13]:


def get_transforms(train=False):
    """
    Takes a list of images and applies the same augmentations to all of them.
    This is completely overengineered but it makes it easier to use in our pipeline
    as drop-in replacement for torchvision transforms.
    ## Example
    ``` python
    imgs = [Image.open(f”image{i}.png”) for i in range(1, 4)]
    t = get_albumentations_transforms(train=True)
    t_imgs = t(imgs) # List[torch.Tensor]
    ```
    For the single image case:
    ```python
    img = Image.open(f”image{0}.png”)
    # or img = np.load(some_bytes)
    t = get_albumentations_transforms(train=True)
    t_img = t(img) # torch.Tensor
    ```
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _data_transform = None
    def _get_transform(n: int = 3):
        if train:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.OneOf(
                        [
                            A.Rotate(limit=0, p=1),
                            A.Rotate(limit=90, p=1),
                            A.Rotate(limit=180, p=1),
                            A.Rotate(limit=270, p=1),
                        ],
                        p=0.5,
                    ),
                    A.Compose(
                        [
                            A.OneOf(
                                [
                                    A.ColorJitter(
                                        brightness=(0.9, 1),
                                        contrast=(0.9, 1),
                                        saturation=(0.9, 1),
                                        hue=(0, 0.1),
                                        p=1.0,
                                    ),
                                    A.Affine(
                                        scale=(0.5, 1.5),
                                        translate_percent=(0.0, 0.0),
                                        shear=(0.5, 1.5),
                                        p=1.0,
                                    ),
                                ],
                                p=0.5,
                            ),
                            A.GaussianBlur(
                                blur_limit=(1, 3), sigma_limit=(0.1, 3), p=1.0
                            ),
                        ]
                    ),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                        ],
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        else:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},

            )
        return data_transforms
    def transform_images(images: any):
        nonlocal _data_transform
        if not isinstance(images, list):
            n = 1
            images = [images]
        else:
            n = len(images)
        if _data_transform is None:
            # instantiate once
            _data_transform = _get_transform(n)
        # accepts both lists of np.Array and PIL.Image
        if isinstance(images[0], Image.Image):
            images = [np.array(img) for img in images]
        image_dict = {"image": images[0]}
        for i in range(1, n):
            image_dict[f"image{i}"] = images[i]
        transformed = _data_transform(**image_dict)
        transformed_images = [
            transformed[key] for key in transformed.keys() if "image" in key
        ]
        if len(transformed_images) == 1:
            return transformed_images[0]
        return transformed_images
    return transform_images


# In[14]:


def load_success_ids(feat_folder: str):
    """
    Backwards-compatible loading of success IDs.
    We either load the available slide ids from the deprecated success.txt file
    or from the success.db sqlite database.
    If both files exist, we always prefer the database.
    """
    success_ids = set()
    success_txt = f"{feat_folder}/success.txt"
    success_db = f"{feat_folder}/success.db"
    if os.path.exists(success_txt):
        print("Warning: Loading success IDs from deprecated success.txt.")
        with open(success_txt, "r") as f:
            for line in f:
                success_ids.add(line.strip())
    if os.path.exists(success_db):
        print("Loading success IDs from database.")
        conn = sqlite3.connect(success_db)
        cursor = conn.cursor()
        cursor.execute("SELECT slide_id FROM success")
        success_ids = set([row[0] for row in cursor.fetchall()])
        conn.close()
    return success_ids


# In[15]:


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# In[16]:


import torch
import numpy as np
import random


# In[17]:


SEED = 42
seed_everything(SEED)


# In[18]:

class IdilDataSet(Dataset):
    """
    Only for single dataset classes!
    """
    def __init__(
        self,
        csv_path: str,
        folder: str,
        magnification: int,
        transform: Optional[Callable] = get_transforms(),
        n_patches: int = 250,
        random_selection=False,
        limit: Optional[int] = None,
        wsi_type: str = "frozen"
    ):
        super().__init__()
        self.csv = pd.read_csv(csv_path)
        self.csv = self.csv[self.csv["wsi_type"] == wsi_type]

        # Filter out unwanted tumor types
        self.csv = self.csv[self.csv["Tumor Type"] != "Oligoastrocytoma"]
        
        # Replace grade values
        self.csv["Neoplasm Histologic Grade"] = self.csv["Neoplasm Histologic Grade"].replace({"G2": "low grade glioma", "G3": "high grade glioma"})
        
        # Replace IDH status values
        self.csv["Subtype"] = self.csv["Subtype"].replace({
            "LGG_IDHmut-non-codel": "IDH mutation",
            "LGG_IDHmut-codel": "IDH mutation",
            "LGG_IDHwt": "wild-type IDH"
        })

        
        self.folder = folder
        self.magnification = magnification
        self.transform = transform
        self.n_patches = n_patches
        self.random_selection = random_selection
        self.slide_ids = self.csv["uuid"].unique()
        success_ids = load_success_ids(self.folder)
        self.slide_ids = [x for x in self.slide_ids if x in success_ids]
        if limit:
            self.slide_ids = self.slide_ids[:limit]
        self.labels = []
        self.patches = []
        self.load_patches()
        self.compute_weights()
        
    def load_patches(self):
        """
        Load n_patches into memory.
        """
        for slide_id in tqdm(self.slide_ids, desc="Prefetch patches"):
            # TODO: adjust `_features.h5` once we renamed it on the storage server
            file = f"{self.folder}/{slide_id}_features.h5"
            try:
                with h5py.File(file, "r") as h5f:
                    n_patches = min(self.n_patches, len(h5f[str(self.magnification)]))
                    # select random indices
                    if self.random_selection:
                        indices = sample(range(n_patches), n_patches)
                    else:
                        indices = list(range(n_patches))
                    imgs = [
                        Image.fromarray(h5f[str(self.magnification)][i]) for i in indices
                    ]
                    self.patches.append((imgs, slide_id))
                    self.labels.append(self.get_label(slide_id))
            except Exception as e:
                pass
    def __len__(self):
        return len(self.patches)
    def get_label(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id, "label"].values[0]
    def get_metadata(self, slide_id):
        return self.csv.loc[self.csv["uuid"] == slide_id]
    def compute_weights(self):
        """
        Compute weights for WeightedRandomSampler.
        """
        class_counts = {}
        for label in self.labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        self.weights = [class_weights[label] for label in self.labels]
        
    def __getitem__(self, idx):
        imgs, slide_id = self.patches[idx]
        #imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack([self.transform(img) for img in imgs])  # Stack images to form a tensor
        label = self.get_label(slide_id)
        metadata = self.get_metadata(slide_id)
        age = metadata["Diagnosis Age"].values[0]
        race = metadata["Race Category"].values[0]
        sex = metadata["Sex"].values[0]
        grade = metadata["Neoplasm Histologic Grade"].values[0]
        IDHstatus = metadata["Subtype"].values[0]
        tumortype = metadata["Tumor Type"].values[0]
        prompt = f"a frozen brain histopathology slide of a {race.lower()}, {sex.lower()}, age {age}, has {IDHstatus.lower()}, {grade.lower()} of {tumortype.lower()}"
        return slide_id, imgs, label, prompt
    


# In[20]:


folder = "/n/data2/hms/dbmi/kyu/lab/che099/data/tcga_lgg/frozen_patches_20x"
csv = "/n/data2/hms/dbmi/kyu/lab/che099/data/idil_tcga_lgg_merge_idh.csv"

assert os.path.exists(folder)
assert os.path.exists(csv)
dataset = IdilDataSet(csv, folder=folder, magnification=20, random_selection=True, limit=None, wsi_type="frozen")


# In[21]:


check_min_version("0.12.0.dev0")


# In[22]:


logger = get_logger(__name__)


# In[26]:


def save_model_card(
    repo_name, images=None, base_model=str, prompt=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"
    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_name}
These are LoRA adaption weights for {repo_name}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


# In[27]:


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# In[28]:


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )
    
    #if '--pretrained_model_name_or_path' not in [arg.dest for arg in parser._actions]:
     #   parser.add_argument('--pretrained_model_name_or_path', type=str, default="stabilityai/stable-diffusion-2-1-base", help='Pretrained model name or path')
    
    #return parser.parse_args()
    return args

# In[29]:


#class DreamBoothDataset(Dataset):
#     """
#     A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
#     It pre-processes the images and the tokenizes prompts.
#     """
#     def __init__(
#         self,
#         instance_data_root,
#         instance_prompt,
#         tokenizer,
#         class_data_root=None,
#         class_prompt=None,
#         size=512,
#         center_crop=False,
#     ):
#         self.size = size
#         self.center_crop = center_crop
#         self.tokenizer = tokenizer

#         # self.instance_data_root = Path(instance_data_root)
#         # if not self.instance_data_root.exists():
#         # raise ValueError("Instance images root doesn't exists.")
#         self.instance_images_path = glob(instance_data_root)
#         print("line427, number of image: {}".format(len(self.instance_images_path)))

#         # self.instance_images_path = list(Path(instance_data_root).iterdir())
#         self.num_instance_images = len(self.instance_images_path)
#         self.instance_prompt = instance_prompt
#         self._length = self.num_instance_images
#         if class_data_root is not None:
#             self.class_data_root = Path(class_data_root)
#             self.class_data_root.mkdir(parents=True, exist_ok=True)
#             self.class_images_path = list(self.class_data_root.iterdir())
#             self.num_class_images = len(self.class_images_path)
#             self._length = max(self.num_class_images, self.num_instance_images)
#             self.class_prompt = class_prompt
#         else:
#             self.class_data_root = None
#         self.image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(
#                     size, interpolation=transforms.InterpolationMode.BILINEAR
#                 ),
#                 (
#                     transforms.CenterCrop(size)
#                     if center_crop
#                     else transforms.RandomCrop(size)
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )
#     def __len__(self):
#         return self._length
#     def __getitem__(self, index):
#         example = {}
#         instance_image = Image.open(
#             self.instance_images_path[index % self.num_instance_images]
#         )
#         if not instance_image.mode == "RGB":
#             instance_image = instance_image.convert("RGB")
#         example["instance_images"] = self.image_transforms(instance_image)
#         example["instance_prompt_ids"] = self.tokenizer(
#             self.instance_prompt,
#             truncation=True,
#             padding="max_length",
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids
#         if self.class_data_root:
#             class_image = Image.open(
#                 self.class_images_path[index % self.num_class_images]
#             )
#             if not class_image.mode == "RGB":
#                 class_image = class_image.convert("RGB")
#             example["class_images"] = self.image_transforms(class_image)
#             example["class_prompt_ids"] = self.tokenizer(
#                 self.class_prompt,
#                 truncation=True,
#                 padding="max_length",
#                 max_length=self.tokenizer.model_max_length,
#                 return_tensors="pt",
#             ).input_ids
#         return example


# In[30]:


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example[3] for example in examples]  # Assuming the prompt is the fourth element
    
    # Extract images from the examples
    pixel_values = [example[1] for example in examples]  # Assuming the images are the second element

    if with_prior_preservation:
        # Handle prior preservation examples if needed 
        input_ids += [example["class_prompt"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    # Stack pixel values into a batch tensor
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # Tokenize input prompts into input_ids
    input_ids = tokenizer(input_ids, padding=True, return_tensors="pt").input_ids

    # Create a batch dictionary
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


# In[31]:


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


# In[32]:


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# In[ ]:


import wandb

def main(args):
    # logging_dir = Path(args.output_dir, "logs")  # Corrected logging directory initialization
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    
    logging_dir = Path(args.output_dir) / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)  

    if args.report_to == "wandb":
        import wandb
        wandb.init(project="my-project", name="experiment-name")
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        #logging_dir=logging_dir,
        #logging_dir=logging_dir if args.report_to in ['tensorboard', 'wandb'] else None
    )

    # if args.report_to == "wandb":
    #    if not wandb_available():
   #          raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
   #     wandb.init(project="your_project_name", config=args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = IdilDataSet(
        csv_path=args.instance_data_dir,
        folder=args.class_data_dir if args.with_prior_preservation else args.instance_data_dir,
        magnification=20,
        transform=get_transforms(train=True),
        n_patches=250,
        random_selection=True,
        limit=None,
        wsi_type="frozen"
    )

    train_dataloader = DataLoader(
        train_dataset,
        #batch_size=args.train_batch_size,
        batch_size=1,
        shuffle=True,
        # collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    
    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(lora_layers)
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-lora", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(accelerator.device)

    # text_encoder.to(accelerator.device)
    projection = None
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Unpack the batch correctly
            slide_ids, imgs, labels, prompts = batch
            # Convert the list of images to tensors
            #pixel_values = torch.stack([img.to(dtype=weight_dtype) for img in imgs])
            #pixel_values = torch.cat([img.unsqueeze(0) for img in imgs], dim=0).to(dtype=weight_dtype)
            #pixel_values = imgs.view(-1, *imgs.shape[2:]).to(dtype=weight_dtype)  # Flatten the first two dimensions
            pixel_values = imgs.view(-1, *imgs.shape[2:]).to(dtype=weight_dtype).to(accelerator.device)
            
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

    
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                # Tokenize the prompts
                inputs = tokenizer(prompts, padding=True, return_tensors="pt")
                input_ids = inputs["input_ids"].to(accelerator.device, dtype=torch.long)  # Corrected dtype to torch.long
                attention_mask = inputs["attention_mask"].to(accelerator.device)  # No need to change dtype for attention_mask
            
                # Get the encoder hidden states
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
                
                if projection is None:
                    batch_size = latents.shape[0]
                    latent_dims = latents.shape[1] * latents.shape[2] * latents.shape[3]
                    projection_output_size = latent_dims
                    projection = nn.Linear(encoder_hidden_states.shape[-1], projection_output_size).to("cuda")
                
                # Flatten encoder_hidden_states before projecting
                encoder_hidden_states_flat = encoder_hidden_states.view(encoder_hidden_states.size(0), -1)
                
                # Project encoder_hidden_states to match latents' total number of elements
                encoder_hidden_states_projected = projection(encoder_hidden_states_flat)

                # Ensure the hidden states have the expected shape
                if encoder_hidden_states.shape != latents.shape:
                    raise ValueError(f"Shape mismatch: encoder_hidden_states.shape ({encoder_hidden_states.shape}) != latents.shape ({latents.shape})")

                encoder_hidden_states_reshaped = encoder_hidden_states_projected.view(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    # Compute instance loss
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(), target_prior.float(), reduction="mean"
                    )
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}."
            )
            # create pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed
            )
            prompt = args.num_validation_images * [args.validation_prompt]
            images = pipeline(
                prompt, num_inference_steps=25, generator=generator
            ).images
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "validation", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "validation": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )
            del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(args.output_dir)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if args.seed
                else None
            )
            prompt = args.num_validation_images * [args.validation_prompt]
            images = pipeline(
                prompt, num_inference_steps=25, generator=generator
            ).images
            test_image_dir = Path(args.output_dir) / "test_images"
            test_image_dir.mkdir()
            for i, image in enumerate(images):
                out_path = test_image_dir / f"image_{i}.png"
                image.save(out_path)
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )
        if args.push_to_hub:
            save_model_card(
                repo_name,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            repo.push_to_hub(
                commit_message="End of training", blocking=False, auto_lfs_prune=True
            )
    accelerator.end_training()



# In[34]:


if __name__ == "__main__":
    args = parse_args()
    main(args)

