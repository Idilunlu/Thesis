# script to generate csv files from WSI files
# replace the BASE and OUTPATH with the appropriate paths

BASE="/n/data2/hms/dbmi/kyu/lab/gw90/datasets/AUS"
OUTPATH="/n/scratch/users/i/idu675"
#
python3 wsi_to_csv.py \
  --base ${BASE} \
  --outpath ${OUTPATH}