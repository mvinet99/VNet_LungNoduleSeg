import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF # Import functional API
from typing import Tuple, List, Dict, Optional
import random # Import random for synchronized transforms

class AllDataset(Dataset):
    def __init__(self, image_dir:str, mask_dir:str,
                 augment:bool=False,
                 normalize:bool=True,
                 mean:List=[-493.0376],
                 std:List=[443.1897]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.normalize = normalize
        
        # Get and sort filenames
        images_list = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        masks_list = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        # --- Validation Step ---
        if len(images_list) != len(masks_list):
            raise ValueError(f"Mismatch in number of files: {len(images_list)} images, {len(masks_list)} masks")
        
        if not images_list: # Check if lists are empty
            raise ValueError(f"Warning: No image files found in {image_dir}")

        for img_file, mask_file in zip(images_list, masks_list):
            # Extract base filename part after the first underscore
            img_base = img_file.split('_', 1)[-1] if '_' in img_file else img_file
            mask_base = mask_file.split('_', 1)[-1] if '_' in mask_file else mask_file
            
            # Compare the base filenames
            if img_base != mask_base:
                raise ValueError(f"Filename mismatch detected after sorting (based on name after first '_'): Image '{img_file}' vs Mask '{mask_file}'")
        # --- End Validation ---

        # Store the validated and sorted lists
        self.images = images_list
        self.masks = masks_list

        # Define normalization transform here if needed
        if self.normalize:
            # Assuming single channel (grayscale). Adjust if multi-channel.
            self.image_normalize = transforms.Normalize(mean=mean, std=std) # Calculated values for LUNA16 train set
        else:
            self.image_normalize = None

    def __len__(self):
        return len(self.images)
    
    def _apply_transforms(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies augmentations and normalization to the image and mask."""
        # --- Apply Augmentations Synchronously to both image and mask --- 
        if self.augment:
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2. Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 3. Random Rotation 
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 4. Random Affine (Translate, Scale, Shear)
            affine_params = transforms.RandomAffine.get_params(degrees=(0,0), translate=(0.1, 0.1), 
                                                               scale_ranges=(0.9, 1.1), shears=(-5, 5), 
                                                               img_size=image.shape[1:]) # Get size from C, H, W
            image = TF.affine(image, *affine_params)
            mask = TF.affine(mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST)
            
        # --- Apply Normalization (Image Only) ---
        if self.image_normalize:
            image = self.image_normalize(image)

        return image, mask

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        # Load image and mask as tensors from .npy files
        try:
            image = torch.tensor(np.load(img_path), dtype=torch.float32)
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32) # Keep as float for Dice
        except Exception as e:
            print(f"Error loading data for index {index}: {e}")
            print(f"Image path: {img_path}, Mask path: {mask_path}")

            # Returning dummy tensors to avoid crashing the loader entirely
            return torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64), dtype=torch.float32)

        # Add channel dimension if image/mask are HxW (assuming grayscale)
        if image.ndim == 2:
            image = image.unsqueeze(0) # Shape: [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)   # Shape: [1, H, W]

        # Apply transformations using the helper method
        image, mask = self._apply_transforms(image, mask)

        return image, mask