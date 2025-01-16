import os
import numpy as np
import cv2
import tifffile as tf
import h5py
import gc  # Garbage collection
from skimage.transform import resize
import argparse
from memory_profiler import profile

@profile
def get_low_res_layer(image_path, downsample_factor=4):
    with tf.TiffFile(image_path) as tif:
        # Access the lowest resolution available
        image = tif.series[-1].asarray()
        # Further downsample for processing to manage memory
        image = resize(image, (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor),
                       anti_aliasing=True)
    return image

@profile
def get_tissue_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 11)
    _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_otsu.astype(np.uint8)

@profile
def extract_useful_patches(tissue_mask, patch_size, threshold):
    useful_patches_coord = []
    for x in range(0, tissue_mask.shape[1], patch_size):
        for y in range(0, tissue_mask.shape[0], patch_size):
            if x + patch_size > tissue_mask.shape[1]:
                x = tissue_mask.shape[1] - patch_size
            if y + patch_size > tissue_mask.shape[0]:
                y = tissue_mask.shape[0] - patch_size
            patch = tissue_mask[y:y + patch_size, x:x + patch_size]
            if np.mean(patch) > threshold:
                useful_patches_coord.append((x, y))
    return useful_patches_coord

@profile
def save_coords_h5(coords, patch_size, h5_path):
    with h5py.File(h5_path, 'w') as file:
        dset = file.create_dataset('coords', data=coords)
        dset.attrs['patch_size'] = patch_size

@profile
def process_image(image_path, save_path, patch_size, threshold):
    img_rgb = get_low_res_layer(image_path)
    tissue_mask = get_tissue_mask(img_rgb)
    coords = extract_useful_patches(tissue_mask, patch_size, threshold)
    h5_path = os.path.join(save_path, os.path.basename(image_path).replace('.ome.tif', '.h5'))
    save_coords_h5(coords, patch_size, h5_path)
    del img_rgb, tissue_mask, coords  # Cleanup
    gc.collect()  # Trigger garbage collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single .ome.tif image to extract tissue patches.")
    parser.add_argument("image_path", type=str, help="Path to the .ome.tif image")
    parser.add_argument("output_dir", type=str, help="Directory to save the output HDF5 file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patch")
    parser.add_argument("--threshold", type=float, default=0.15, help="Threshold for selecting useful patches")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing image: {args.image_path}")
    process_image(args.image_path, args.output_dir, args.patch_size, args.threshold)
    print(f"Output saved to {args.output_dir}")
