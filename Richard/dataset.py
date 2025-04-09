import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF # Import functional API
from typing import Tuple, List, Dict, Optional
import random # Import random for synchronized transforms

def ensure_tensor_size(tensor:torch.Tensor, target_size:Tuple[int,int,int]=(64, 64, 64)):
    # Check if the tensor is already the desired size
    if tensor.shape[1:] == target_size:  
        return tensor

    # Add a batch dimension 
    tensor = tensor.unsqueeze(0)

    # Permute the tensor 
    tensor = tensor.permute(0, 1, 4, 2, 3)

    # Resize using trilinear interpolation
    resized_tensor:torch.Tensor = F.interpolate(
        tensor.float(), 
        size=target_size, 
        mode='trilinear', 
        align_corners=False
    )

    # Remove batch dimension and permute back 
    return resized_tensor.squeeze(0).permute(0, 2, 1, 3) 

def ensure_tensor_size_mask(tensor:torch.Tensor, target_size:Tuple[int,int,int]=(64, 64, 64)):
    # Ensure tensor is at least 3D
    if tensor.dim() == 3:  
        tensor = tensor.unsqueeze(0)  # Add a channel dimension 

    # Ensure tensor is 4D 
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)  # Add a batch dimension

    # Ensure correct shape before resizing
    if tensor.shape[2:] != target_size:  
        tensor = F.interpolate(
            tensor.float(),
            size=target_size,
            mode="nearest"  
        )

    return tensor.squeeze(0).squeeze(0)  # Remove batch dimension

class AllDataset(Dataset):
    def __init__(self, image_dir:str, mask_dir:str,
                 augment:bool=False,
                 normalize:bool=True): # Added normalize flag
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
            # These values might need adjustment based on your dataset's statistics.
            self.image_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        else:
            self.image_normalize = None

    def __len__(self):
        return len(self.images)
    
    def _apply_transforms(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies augmentations and normalization to the image and mask."""
        # --- Apply Augmentations Synchronously --- 
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
            # Use nearest neighbor interpolation for masks to avoid creating new pixel values
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 4. Random Affine (Translate, Scale, Shear)
            # Get parameters first, then apply to both
            affine_params = transforms.RandomAffine.get_params(degrees=(0,0), translate=(0.1, 0.1), 
                                                               scale_ranges=(0.9, 1.1), shears=(-5, 5), 
                                                               img_size=image.shape[1:]) # Get size from C, H, W
            image = TF.affine(image, *affine_params)
            mask = TF.affine(mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST)
            
        # --- Apply Normalization (Image Only) ---
        if self.image_normalize:
            image = self.image_normalize(image)

        # Ensure mask remains in [0, 1] range after potential float transforms if needed
        # mask = torch.clamp(mask, 0, 1)

        # Ensure final mask is the correct type (float for Dice/BCE, long for CrossEntropy)
        # Keeping float32 as per previous assumption

        return image, mask

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        
        # Load image and mask as tensors (assuming they are saved as numpy arrays)
        # Ensure they are loaded correctly, potentially add error handling
        try:
            image = torch.tensor(np.load(img_path), dtype=torch.float32)
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32) # Keep as float for Dice
        except Exception as e:
            print(f"Error loading data for index {index}: {e}")
            print(f"Image path: {img_path}, Mask path: {mask_path}")
            # Return None or raise error, depending on desired behavior
            # Returning dummy tensors to avoid crashing the loader entirely
            # Adjust size as needed
            return torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64), dtype=torch.float32)

        # Add channel dimension if image/mask are HxW (assuming grayscale)
        if image.ndim == 2:
            image = image.unsqueeze(0) # Shape: [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)   # Shape: [1, H, W]

        # Apply transformations using the helper method
        image, mask = self._apply_transforms(image, mask)

        return image, mask