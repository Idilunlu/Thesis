# simple script to convert a directory of WSIs to a CSV file
# the script will search for WSIs in the directory and its subdirectories
# the CSV file will contain the paths to the WSIs
# and will create a CSV file for each directory containing WSIs so that the CSV file can be used for batch processing

import glob
import os
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

def process_directory(base, outpath, directory_name):
    full_path = os.path.join(base, directory_name)
    print("Full path being accessed:", full_path)  # Debug print
    try:
        # please replace ".ndpi" with the extension of the images you are working with
        WSIs = glob.glob(os.path.join(full_path, '**/*.ome.tif'), recursive=True)
        print(f"Number of .ndpi files found: {len(WSIs)}")  # Debug print
        if len(WSIs) == 0:
            print("No files found, checking directory contents directly:")
            print(os.listdir(full_path))  # List contents if no files are found
        paths = [{'WSI_path': WSI} for WSI in WSIs]
        df = pd.DataFrame(paths)
        outfile = os.path.join(outpath, f"{directory_name}.csv")
        df.to_csv(outfile, index=False)
        print(f"Processed {len(WSIs)} WSIs in {directory_name}, output file: {outfile}")
    except Exception as e:
        print(f"Error accessing files in {full_path}: {e}")  # Print any error

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base", help="Base input directory", type=str, required=True)
    parser.add_argument("--outpath", help="Output directory", type=str, required=True)
    args = parser.parse_args()

    # please replace "images" with the name of the directory containing the images 
    directories_to_process = ["images"]
    for directory in directories_to_process:
        process_directory(args.base, args.outpath, directory)
