# Description: This script processes whole-slide images to extract useful patches.
# The script takes a CSV file containing the paths to the WSIs and processes them
# to extract useful patches. The script saves the coordinates of the useful patches
# in an HDF5 file and also saves a visual mask of the tissue regions in the WSI.
# The script uses OpenSlide to read the WSIs and SimpleITK to read the OME-TIFF files.

import os
import numpy as np
import cv2
import openslide
import SimpleITK
import h5py
import pandas as pd
from skimage.filters import threshold_otsu
import argparse
import matplotlib.pyplot as plt

# Define function to visualize images
def display_image(image, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compute_patch_size(wsi, target_mpp, target_patch_size, downsample_rate, mpp=None):
    if mpp is None:
        if wsi.properties.get("openslide.mpp-x"):
            mpp = float(wsi.properties.get("openslide.mpp-x", 1))
            print("#1 mpp:", mpp)
        else:
            unit = wsi.properties.get("tiff.ResolutionUnit")
            x_resolution = float(wsi.properties.get("tiff.XResolution"))
            if unit.lower() == "centimeter":
                mpp = 10000 / x_resolution
            else:
                mpp = 25400 / x_resolution
            print("#2 mpp:", mpp)

        print("# mpp = ", mpp)
        if mpp > 1:
            level0_size = int(target_mpp / float(mpp) * 2 * target_patch_size / 2)
        else:
            level0_size = int(round(target_mpp / float(mpp)) / 2 * target_patch_size * 2)

        print("level0_size", level0_size)

    level0_size = round(target_mpp / float(mpp) / 2) * target_patch_size * 2
    print(level0_size)

    return level0_size // downsample_rate

def save_coords_h5(coords, patch_size, h5_path):
    file = h5py.File(h5_path, 'w')
    dset = file.create_dataset('coords', data=coords)
    dset.attrs['patch_size'] = patch_size
    file.close()

def save_coords_h5_modified(coords, patch_size, h5_path, original_path):
    file = h5py.File(h5_path, 'w')
    dset = file.create_dataset('coords', data=coords)
    dset.attrs['patch_size'] = patch_size
    dset.attrs['path'] = original_path
    file.close()

def get_thumbnail(wsi, downsample=16):
    full_size = wsi.dimensions
    img_rgb = np.array(wsi.get_thumbnail((int(full_size[0] / downsample), int(full_size[1] / downsample))))
    return img_rgb

def get_thumbnail_png(wsi, downsample=16):
    width = int(wsi.shape[1] // downsample)
    height = int(wsi.shape[0] // downsample)
    dim = (width, height)
    img_rgb = cv2.resize(wsi, dim, interpolation=cv2.INTER_AREA)
    return img_rgb

def get_tissue_mask(img_RGB):
    img_RGB[np.all(img_RGB <= (50, 50, 50), axis=-1)] = (255, 255, 255)
    img = img_RGB
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 11)
    _, img_otsu = cv2.threshold(img_med, 15, 255, cv2.THRESH_BINARY)
    tissue_mask = img_otsu.astype(np.uint8)
    return tissue_mask

def extract_useful_patches(tissue_mask, patch_size, threshold):
    useful_patches_coord = []
    for x in np.arange(0, tissue_mask.shape[1], patch_size):
        for y in np.arange(0, tissue_mask.shape[0], patch_size):
            if (x + patch_size > tissue_mask.shape[1]):
                x = tissue_mask.shape[1] - patch_size
            if (y + patch_size > tissue_mask.shape[0]):
                y = tissue_mask.shape[0] - patch_size
            patch = tissue_mask[y: y + patch_size, x: x + patch_size]
            if patch.mean() > threshold:
                useful_patches_coord.append([x, y])
    return useful_patches_coord

def save_visual_mask(img_rgb, tissue_mask, visual_mask_path):
    overlay = img_rgb.copy()
    overlay[tissue_mask == 0] = (0, 0, 0)
    cv2.imwrite(visual_mask_path, overlay)

def compute_coords_single(wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args):
    slide_id = '.'.join(os.path.basename(wsi_path).split('.')[:-1])
    patch_coord_h5 = os.path.join(patch_coord_dir, slide_id + '.h5')
    visual_mask = os.path.join(visual_mask_dir, slide_id + '.jpg')

    if args.set_png:
        wsi_image = SimpleITK.ReadImage(wsi_path)
        wsi = SimpleITK.GetArrayFromImage(wsi_image)
        img_rgb = get_thumbnail_png(wsi, downsample=args.downsample)
        tissue_mask = get_tissue_mask(img_rgb)
    else:
        wsi = openslide.open_slide(wsi_path)
        img_rgb = get_thumbnail(wsi, downsample=args.downsample)
        tissue_mask = get_tissue_mask(img_rgb)

    save_visual_mask(img_rgb, tissue_mask, visual_mask)
    # display_image(img_rgb, "RGB Thumbnail")
    # display_image(tissue_mask, "Tissue Mask")

    if args.set_mpp:
        patch_size_downsample = compute_patch_size(wsi, args.target_mpp, args.patch_size, args.downsample, args.mpp)
    else:
        patch_size_downsample = compute_patch_size(wsi, args.target_mpp, args.patch_size, args.downsample)

    coords_downsample = extract_useful_patches(tissue_mask, patch_size_downsample, args.threshold)
    coords = np.array(coords_downsample) * args.downsample
    save_coords_h5_modified(coords, patch_size_downsample * args.downsample, patch_coord_h5, wsi_path)

def get_result_dirs(result_root):
    patch_coord_dir = os.path.join(result_root, 'patch_coord')
    patch_feature_dir = os.path.join(result_root, 'patch_feature')
    visual_mask_dir = os.path.join(result_root, 'visual_mask')
    visual_stitch_dir = os.path.join(result_root, 'visual_stitch')

    if not os.path.isdir(patch_coord_dir):
        os.makedirs(patch_coord_dir)
    if not os.path.isdir(patch_feature_dir):
        os.makedirs(patch_feature_dir)
    if not os.path.isdir(visual_mask_dir):
        os.makedirs(visual_mask_dir)
    if not os.path.isdir(visual_stitch_dir):
        os.makedirs(visual_stitch_dir)

    return patch_coord_dir, patch_feature_dir, visual_mask_dir, visual_stitch_dir
    
def process_all_ws_images(wsi_path_list, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args):
    for index, wsi_path in enumerate(wsi_path_list):
        try:
            print(f"Processing {index+1}/{len(wsi_path_list)}: {wsi_path}")
            compute_coords_single(wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args)
            print(f"Successfully processed: {wsi_path}")
        except Exception as e:
            print(f"Failed to process {wsi_path}: {e}")


def get_args():
    parser = argparse.ArgumentParser(description='Process Whole-Slide Images.')
    parser.add_argument('--target_mpp', type=float, default=1, help='Target microns per pixel.')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of the patch.')
    parser.add_argument('--downsample', type=int, default=16, help='Downsample rate for processing.')
    parser.add_argument('--threshold', type=float, default=0.15, help='Threshold for patch selection.')
    parser.add_argument('--num_worker', type=int, default=20, help='Number of workers for parallel processing.')
    parser.add_argument('--save_root', type=str, help='Root directory for saving results.')
    parser.add_argument('--wsi_path', type=str, help='Path to the WSI files.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing WSI paths.')
    parser.add_argument('--n_part', type=int, default=1, help='Total parts to divide the dataset.')
    parser.add_argument('--part', type=int, default=0, help='Specific part to process.')
    parser.add_argument('--column', type=str, default='WSI_path', help='Column name in the CSV where WSI paths are stored.')
    return parser.parse_args()

#args = Args()
args = get_args()

# Get WSI path list
df = pd.read_csv(args.csv_path)
wsi_path_list = list(df['WSI_path'].values)
# print(wsi_path_list)
# Create storage space
patch_coord_dir, patch_feature_dir, visual_mask_dir, visual_stitch_dir = get_result_dirs(args.save_root)

# Process a single WSI file
# compute_coords_single(wsi_path_list[0], patch_coord_dir, visual_mask_dir, visual_stitch_dir, args)
process_all_ws_images(wsi_path_list, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args)
