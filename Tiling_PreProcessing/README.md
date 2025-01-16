# Processing
- `get_csv.sh` : needs to be adjusted to where you want to store the files, given the path to dataset that includes .ndpi (or other supported formats) files it produces csv tables for other modules to iterate over
- `tiling.sh` : reads the created csv files and processes them (diving the slides into patches) and then save the center coordinates of the patches into given folder
- `run_normalization.sh`(optional) : uses normalization.py and macenko_mod.py. They should be in the same folder
- `array2_extract_CHIEF.sh` : advised to assign each file to an individiual slurm jobs for parallel processing
  - it uses `feature_CHIEF2.py`
- `UNI2.sh`
  - it uses `feature_UNI.py` which uses `transforms.py`for the normalization process
 
##### When the data is in .tif format instead of .ndpi , processing steps has been changed slightly.
The features were extracted from the first layer of the pyramidal structure of .tif files.
    
### Attention: For the CHIEF feature extractor version of timm-0.5.4 is required whereas for UNI you need a new version (timm-0.9.8 and higher)

