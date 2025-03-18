import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

# Define a custom dataset class
class ZeroDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_shape=(1, 50, 50, 28)):
        """
        num_samples: number of samples in the dataset
        input_shape: shape of each input sample
        """
        self.num_samples = num_samples
        self.input_shape = input_shape

    def __len__(self):
        # Return the total number of samples
        return self.num_samples

    def __getitem__(self, idx):
        # Return a zero-filled tensor of the input shape and a dummy label (0)
        input_tensor = torch.zeros(self.input_shape)
        mask = torch.zeros((2, 96, 96, 16))  # The segmentation mask with the same shape as the input
        return input_tensor, mask

def ensure_tensor_size(tensor, target_size=(96, 96, 32)):
    # Check if the tensor is already the desired size
    if tensor.shape[1:] == target_size:  
        return tensor  # Already the correct size

    # Add a batch dimension (unsqueeze) - (C, H, W, D) → (1, C, H, W, D)
    tensor = tensor.unsqueeze(0)

    # Permute the tensor from (N, C, H, W, D) to (N, C, D, H, W)
    tensor = tensor.permute(0, 1, 4, 2, 3)  # (N, C, H, W, D) -> (N, C, D, H, W)

    # Resize using trilinear interpolation
    resized_tensor = F.interpolate(
        tensor.float(), 
        size=target_size, 
        mode='trilinear', 
        align_corners=False
    )

    # Remove batch dimension and permute back to (C, H, W, D)
    return resized_tensor.squeeze(0).permute(0, 2, 1, 3)  # (N, C, D, H, W) -> (C, H, W, D)


def ensure_tensor_size_mask(tensor, target_size=(96, 96, 32)):
    # Ensure tensor is at least 3D
    if tensor.dim() == 3:  
        tensor = tensor.unsqueeze(0)  # Add a channel dimension (C, H, W, D)

    # Ensure tensor is 4D (N, C, D, H, W)
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)  # Add a batch dimension

    # Ensure correct shape before resizing
    if tensor.shape[2:] != target_size:  
        tensor = F.interpolate(
            tensor.float(),
            size=target_size,
            mode="nearest"  # Removed align_corners
        )

    return tensor.squeeze(0).squeeze(0)  # Remove batch dimension

class AllDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, diam_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.diam_dir = diam_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        diam_path = os.path.join(self.diam_dir, self.images[index])
        image = torch.unsqueeze(torch.tensor(np.load(img_path)), 0)
        image = ensure_tensor_size(image)
        
        mask = torch.tensor(np.load(mask_path))
        mask = ensure_tensor_size_mask(mask)

        diam = torch.tensor(np.load(diam_path))
        
        return image, mask, diam
    