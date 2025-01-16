# this script is used to normalize the images using Macenko's method
 

import torch
from torchvision import transforms
import openslide
import cv2
import numpy as np  # Ensure numpy is imported
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import os
from os.path import basename, dirname
from macenko_mod import TorchMacenkoNormalizer
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--csv_file", type=str,
                    help="csv file containing images to normalize", default='data_tile_csv/TCGA_GBM_PCNSL_FS_top500_labels.csv')
parser.add_argument("--outpath", type=str, help="Output path",
                    default='/n/scratch3/users/s/shl968/TCGA_GBM_tiles_normalized_nofit/FS_TILES')
parser.add_argument("--target", type=str, help="Target img for normalization",
                    default=None)
parser.add_argument("--i_start", type=int, help="start", default=0)
parser.add_argument("--i_end", type=int, help="end", default=10000000)
parser.add_argument("--Io_source", type=int, help="Io", default=250)
parser.add_argument("--Io_target", type=int, help="Io", default=250)
parser.add_argument("--beta", type=int, help="end", default=0.15)
parser.add_argument("--debug", help="end",
                    action='store_true', default=False)

args = parser.parse_args()

def read_ndpi_image(file_path):
    try:
        slide = openslide.OpenSlide(file_path)
        # Read the entire image at the lowest resolution
        img = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

if __name__ == "__main__":
    args_dict = vars(args)
    print('============   parameters   ============')
    for key in args_dict.keys():
        print(f'{key}:\t{args_dict[key]}')
    print('========================================')
    df = pd.read_csv(args.csv_file)
    args.i_end = min(args.i_end, df.shape[0])
    df = df.loc[args.i_start:args.i_end]

    source_files = list(df['WSI_path'])  # Ensure the column name matches your CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    torch_normalizer = TorchMacenkoNormalizer()

    if args.target is not None:
        target_img = cv2.cvtColor(cv2.imread(args.target), cv2.COLOR_BGR2RGB)
        torch_normalizer.fit(T(target_img).to(
            device), Io=args.Io_target, beta=args.beta, upper=True)

    failed_files = []
    for source_file in tqdm(source_files, mininterval=10):
        filename = basename(source_file)
        folder_name = basename(dirname(source_file))
        exist = []

        for id in ['norm']:
            outpath = os.path.join(args.outpath, id, folder_name)
            outname = os.path.join(outpath, filename)
            exist.append(os.path.isfile(outname))
        if all(exist):
            continue

        try:
            source_img = read_ndpi_image(source_file)
            if source_img is None:
                failed_files.append(source_file)
                continue

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            t_source = T(source_img).to(device)
            norm, H, E = torch_normalizer.normalize(
                I=t_source, stains=True, Io=args.Io_source, Io_out=args.Io_target, beta=args.beta)
        except Exception as e:
            print(f"Error processing {source_file}: {e}")
            failed_files.append(source_file)
            continue

        if args.debug:
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(1, 2, 1)
            plt.imshow(target_img)
            ax = plt.subplot(1, 2, 2)
            plt.imshow(norm)
            plt.savefig('test.jpg')

        for img, id in zip([norm], ['norm']):
            outpath = os.path.join(args.outpath, id, folder_name)
            outname = os.path.join(outpath, filename)
            os.makedirs(outpath, exist_ok=True)
            cv2.imwrite(outname, img.cpu().numpy())
            print(f"Saved {outname}")

        del source_img, t_source, norm, H, E

    df_failed = pd.DataFrame({'failed_files': failed_files})
    df_failed.to_csv('failed_PM_files.csv')
