import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Richard.src.data.dataset import AllDataset  

# --- Configuration ---
# IMPORTANT: Replace these with the actual paths to your TRAINING dataset
TRAIN_IMAGE_DIR = "/radraid2/dongwoolee/VNet_LungNoduleSeg/data/LUNA16/train/images_2D_0axis"
TRAIN_MASK_DIR = "/radraid2/dongwoolee/VNet_LungNoduleSeg/data/LUNA16/train/masks_2D_0axis" # Mask dir needed for dataset init, but masks aren't used for stats
BATCH_SIZE = 16 
NUM_WORKERS = 4
# --- End Configuration ---

def calculate_mean_std(image_dir: str, mask_dir: str, batch_size: int, num_workers: int) -> tuple[float, float]:
    """
    Calculates the mean and standard deviation of pixel values across a dataset.

    Args:
        image_dir: Path to the image directory.
        mask_dir: Path to the mask directory.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of workers for the DataLoader.

    Returns:
        A tuple containing the mean and standard deviation.
    """
    print(f"Calculating stats for images in: {image_dir}")

    # Ensure paths exist (basic check)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
         # Masks aren't used for calculation, but AllDataset requires the dir
        print(f"Warning: Mask directory not found: {mask_dir}. Proceeding as AllDataset might handle it.")
        # Depending on how critical mask existence is during dataset init,
        # you might need to create dummy masks or adjust AllDataset.


    # Instantiate dataset without normalization or augmentation
    try:
        dataset = AllDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            augment=False,
            normalize=False  # Crucial: calculate stats on original pixel values
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        print("Please ensure image and mask directories contain matching files.")
        return (0.0, 0.0) # Return dummy values or raise

    if len(dataset) == 0:
        print("Dataset is empty. Cannot calculate stats.")
        return (0.0, 0.0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False # No need to shuffle for stats calculation
    )

    # Variables for calculation
    channel_sum = 0.
    channel_sum_sq = 0.
    pixel_count = 0

    num_batches = len(loader)
    print(f"Iterating through {len(dataset)} images in {num_batches} batches...")

    mask_values =[]
    # Wrap loader with tqdm for progress bar
    for i, (images, masks) in enumerate(tqdm(loader, desc="Calculating Stats", unit="batch")):
        # images shape: [B, C, H, W] (assuming AllDataset returns this)
        # Ensure images are float (AllDataset seems to do this)
        # If images were loaded as uint8 (0-255), you would need to scale:
        # images = images.float() / 255.0

        # Flatten C, H, W dimensions for easier calculation per channel
        # Assuming single channel (C=1) based on previous Normalize([0.5], [0.5])
        # If multi-channel, this needs adjustment
        if images.shape[1] != 1:
            print(f"Warning: Expected single-channel images, but got shape {images.shape}. Calculating stats for the first channel only.")
            # Or adapt to calculate per channel if needed

        # Calculate sum and sum of squares for the batch
        # Keep batch dimension for now
        # Sum over H, W dimensions: result shape [B, C]
        sum_over_hw = torch.sum(images, dim=[2, 3])
        sum_sq_over_hw = torch.sum(images ** 2, dim=[2, 3])

        # Sum over batch dimension: result shape [C]
        channel_sum += torch.sum(sum_over_hw, dim=0)
        channel_sum_sq += torch.sum(sum_sq_over_hw, dim=0)

        # Count pixels: B * C * H * W
        # Correct way: Count pixels contributing to the sum
        num_pixels_in_batch = images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3]
        pixel_count += num_pixels_in_batch


        # Flatten the masks to get all values
        flat_masks = masks.flatten()
        # Get unique values and add them to the list
        unique_values = torch.unique(flat_masks).tolist()
        for val in unique_values:
            if val not in mask_values:
                mask_values.append(val)



    if pixel_count == 0:
        print("No pixels processed. Check dataset and image loading.")
        return (0.0, 0.0)

    # Calculate mean and std
    # mean = E[X]
    mean = channel_sum / pixel_count
    # var = E[X^2] - (E[X])^2
    variance = (channel_sum_sq / pixel_count) - (mean ** 2)
    # std = sqrt(var)
    # Add epsilon for numerical stability if variance is close to zero
    epsilon = 1e-5
    std_dev = torch.sqrt(variance + epsilon)

    # Assuming single channel, return float values
    # If multi-channel, you'd return the tensors `mean` and `std_dev`
    mean_val = mean.item() if mean.numel() == 1 else mean.tolist()
    std_dev_val = std_dev.item() if std_dev.numel() == 1 else std_dev.tolist()

    print("-" * 30)
    print(f"Calculation Complete.")
    print(f"Total Pixels Processed: {pixel_count}")
    print(f"Calculated Mean: {mean_val}")
    print(f"Calculated Std Dev: {std_dev_val}")
    print(f"Unique Mask Values: {mask_values}")
    print("-" * 30)

    return mean_val, std_dev_val

if __name__ == "__main__":
    # Basic check if paths are default placeholders
    if "path/to/" in TRAIN_IMAGE_DIR or "path/to/" in TRAIN_MASK_DIR:
        print("="*50)
        print("ERROR: Please update TRAIN_IMAGE_DIR and TRAIN_MASK_DIR in the script")
        print("       with the correct paths to your training dataset.")
        print("="*50)
    else:
        calculated_mean, calculated_std = calculate_mean_std(
            image_dir=TRAIN_IMAGE_DIR,
            mask_dir=TRAIN_MASK_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        print("Suggestion:")
        print("Update your dataset's normalization transform with these values:")
        print(f"self.image_normalize = transforms.Normalize(mean=[{calculated_mean:.4f}], std=[{calculated_std:.4f}])") 