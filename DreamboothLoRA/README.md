# Dreambooth-LoRA

Dreambooth-LoRA is a project aimed at creating synthetic histopathology slides in a parameter-efficient way.

---

## Data Preprocessing

Use **`h5_to_png.ipynb`** to convert H5 datasets into PNG format and preprocess the images. Ensure the resolution meets the required standards for training and evaluation.

---

## Model Training

Run **`run_XYZ_prompt.sh`** to train the model. Configure the following parameters in the script:

- **`MODELNAME`**: Path to a compatible HuggingFace model (supports Stable Diffusion 2.1).
- **`CLASS_DATA_DIR`**: (Optional) Directory for class data, used for prior-preservation loss.
- **`-class_prompt`**: (Optional) Generic prompt for the class. It is recommended that the class prompt is broader than the instance prompt, as per DreamBooth guidelines.
- **`-instance_prompt`**: (Mandatory) Specific prompt for generating instances.
- **Hyperparameters**: Adjust the training settings, such as learning rate and batch size.
- **Validation Options**: If validation during training is desired:
  - `validation_epochs`: Set the interval (number of epochs) for validation checks.
  - `-validation_prompt`: Provide a validation prompt for generating outputs during training.

---

## File Descriptions

- **`h5_to_png.ipynb`**: Converts H5 data into PNG format for preprocessing.
- **`run_XYZ_prompt.sh`**: Shell script for running the training pipeline with specified parameters.
- **`Image_gen.py`**: Handles final image generation using the trained model weights.
- **`remove_corrupt.ipynb`**: Removes corrupted files from the output directory before evaluation.
- **`Metrics/`**: Contains notebooks for calculating evaluation metrics:
  - `FID.ipynb`: Calculates Frechet Inception Distance (FID) score.
  - `PSNR.ipynb`: Computes Peak Signal-to-Noise Ratio (PSNR).
  - `SSIM.ipynb`: Measures Structural Similarity Index (SSIM).

---

## Additional Notes

- Ensure the `CLASS_DATA_DIR` and prompts are configured correctly for your dataset to optimize training results.
- For best results, run **`remove_corrupt.ipynb`** after training to clean up the output directory before calculating metrics.
- Metrics notebooks require two input folders: one for real images and one for generated images.


