# Synthetic Data Generation for Improved Brain Histopathology Diagnosis Using Generative Models

This repository contains the code and resources used for the master's thesis titled **"Synthetic Data Generation for Improved Brain Histopathology Diagnosis Using Generative Models"**. The project leverages diffusion models, DreamBooth, and Low-Rank Adaptation (LoRA) to generate synthetic data for brain histopathology and improve downstream diagnostic tasks.

---

## Folder Structure

- **`DreamboothLoRA/`**: Contains scripts and resources for fine-tuning diffusion models using DreamBooth and LoRA and the detailed expalantion of the parameters in shell scripts
- **`Metrics/`**: Includes notebooks to evaluate the quality of generated images using metrics like FID, PSNR, and SSIM.
- **`Tiling_PreProcessing/`**: Contains preprocessing scripts for tiling whole-slide images (WSIs) and extracting regions of interest and alternatively features.
- **`Upscalers/`**: Includes scripts and models for image super-resolution and upscaling synthetic patches.
- **`Distribution_plotter.ipynb`**: Visualizes data distributions for class imbalance analysis.


### Model Training
1. Fine-tune the diffusion model using the scripts in `DreamboothLoRA/`.
2. Configure hyperparameters such as `MODELNAME`, `-instance_prompt`, and `-class_prompt`.
3. Use `Image_gen.py` to generate synthetic patches after training.
4. For multi-magnification outputs, run the upscaling scripts in `Upscalers/`.


### Evaluation
1. Evaluate generated images using notebooks in `Metrics/`.
2. Analyze metrics to compare synthetic data quality with real data.


## Results

The project demonstrates that:
1. Synthetic data augmentation improves classification accuracy for imbalanced tasks like IDH mutation status prediction and tumor grading.
2. Weighted prompting and upscaling with DDPM resulted in successful generated images.

