import numpy as np
import os
import torch
import torch.nn.functional as F

def ensure_tensor_size(tensor, target_size=(64, 64, 64)):
    # Check if the tensor is already the desired size
    if tensor.shape[1:] == target_size:  
        return tensor

    # Add a batch dimension 
    tensor = tensor.unsqueeze(0)

    # Permute the tensor 
    tensor = tensor.permute(0, 1, 4, 2, 3)

    # Resize using trilinear interpolation
    resized_tensor = F.interpolate(
        tensor.float(), 
        size=target_size, 
        mode='trilinear', 
        align_corners=False
    )

    # Remove batch dimension and permute back 
    return resized_tensor.squeeze(0).permute(0, 2, 1, 3) 


def ensure_tensor_size_mask(tensor, target_size=(64, 64, 64)):
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
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        #diam_path = os.path.join(self.diam_dir, self.images[index])
        image = torch.unsqueeze(torch.tensor(np.load(img_path)), 0)
        #image = ensure_tensor_size(image)
        
        mask = torch.tensor(np.load(mask_path))
        #mask = ensure_tensor_size_mask(mask)

        #diam = torch.tensor(np.load(diam_path))
        
        return image, mask#, diam
    