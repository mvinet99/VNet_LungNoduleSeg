import numpy as np
import os
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict

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

class AllDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir): # ,diam_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        #self.diam_dir = diam_dir
        
        # Get and sort filenames
        images_list = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        masks_list = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        # --- Validation Step ---
        if len(images_list) != len(masks_list):
            raise ValueError(f"Mismatch in number of files: {len(images_list)} images, {len(masks_list)} masks")
        
        if not images_list: # Check if lists are empty
             print(f"Warning: No image files found in {image_dir}")
             # Depending on requirements, you might want to raise an error here instead

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

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        
        # Load image and mask, ensuring they are float32 tensors
        image = torch.tensor(np.load(img_path), dtype=torch.float32)
        mask = torch.tensor(np.load(mask_path), dtype=torch.float32)

        # Add channel dimension (assuming grayscale)
        image = image.unsqueeze(0)
        # Masks typically don't need a channel dim for loss calculation, 
        # but might need one depending on the loss function or model. 
        # If your model/loss expects (N, C, D, H, W) for masks, uncomment the next line:
        # mask = mask.unsqueeze(0) 
        
        return image, mask
    