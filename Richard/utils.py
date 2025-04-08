import os
import torch
import numpy as np
import statistics
import random
from model import VNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
                  
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice

class DiameterLoss(nn.Module):
    def __init__(self):
        super(DiameterLoss, self).__init__()

    def compute_weighted_centroid(self, mask):
        """ Compute soft centroid as a weighted sum of coordinates. """
        B, H, W, D = mask.shape  # (Batch, Height, Width, Depth)
        device = mask.device

        # Generate coordinate grids
        y_coords = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)
        x_coords = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)

        # Convert to same dtype as mask
        y_coords = y_coords.to(mask.dtype)
        x_coords = x_coords.to(mask.dtype)

        # Compute weighted centroid
        total_mass = mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-6  # Avoid div by zero
        centroid_x = (x_coords * mask).sum(dim=(1, 2, 3), keepdim=True) / total_mass
        centroid_y = (y_coords * mask).sum(dim=(1, 2, 3), keepdim=True) / total_mass

        return centroid_x.view(B), centroid_y.view(B)  # Ensure correct shape

    def compute_diameter(self, mask):
        """ Compute a differentiable approximation of the longest diameter. """
        B, H, W, D = mask.shape  # (Batch, Height, Width, Depth)
        device = mask.device

        # Generate coordinate grids
        y_coords = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)
        x_coords = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)

        # Convert to same dtype as mask
        y_coords = y_coords.to(mask.dtype)
        x_coords = x_coords.to(mask.dtype)

        # Compute centroid
        centroid_x, centroid_y = self.compute_weighted_centroid(mask)

        # Reshape centroid for broadcasting
        centroid_x = centroid_x.view(B, 1, 1, 1)  # Shape (B,1,1,1)
        centroid_y = centroid_y.view(B, 1, 1, 1)  # Shape (B,1,1,1)

        # Compute distances from centroid
        dist_x = (x_coords - centroid_x).abs()
        dist_y = (y_coords - centroid_y).abs()

        # Approximate max distance in x and y directions (soft max)
        max_x = torch.sum(dist_x * mask, dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
        max_y = torch.sum(dist_y * mask, dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)

        # Approximate max diameter using Pythagorean theorem
        soft_diameter = torch.sqrt(max_x**2 + max_y**2)  # Shape (B,)
        return soft_diameter

    def forward(self, inputs, targets):
        """ Compute differentiable loss as absolute diameter difference. """
        inputs = torch.sigmoid(inputs)  # Ensure values are in (0,1) range

        # Extract only the segmentation mask (assumes first channel contains mask)
        inputs = inputs[:, 0, :, :, :]  # Shape (16, 96, 96, 32)

        # Compute predicted diameter
        pred_diameter = self.compute_diameter(inputs)  # Shape (16,)

        # Compute loss (absolute difference with targets)
        loss = torch.abs(pred_diameter - targets).mean()
        return loss


# Make predictions on val / test set and evaluate average dice score 

def set_seed(seed: int):
    """Set random number generator seeds for reproducibility.

    This function sets seeds for the random number generators in NumPy, Python's
    built-in random module, and PyTorch to ensure that random operations are
    reproducible. 

    Args:
        seed (int): The seed value to use for setting random number generator seeds.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Dice coefficient calculation
def dice_score(pred:torch.Tensor, target:torch.Tensor):
    # Remove thresholding as pred now contains class indices (0 or 1)
    # Ensure both tensors are float for multiplication
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target)
    # Add small epsilon to avoid division by zero
    return (2. * intersection) / (torch.sum(pred) + torch.sum(target) + 1e-6)

# Load the model
def load_model(checkpoint_path:os.PathLike, device:torch.device, eval:bool=True):
    model = VNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    if eval:
        model.eval()
    return model

# Predict and save masks
def predict_and_evaluate(model:torch.nn.Module, image_folder:os.PathLike, mask_folder:os.PathLike, output_folder:os.PathLike, device:torch.device):
    dice_scores = []
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.npy')])
    
    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename).replace('images', 'masks')
        output_path = os.path.join(output_folder, filename)

        # Load NumPy image
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
       
        with torch.no_grad():
            pred_mask = model(image)
            pred_mask = pred_mask.squeeze().cpu().numpy()

        np.save(output_path, pred_mask)

        # Compute Dice score
        gt_mask = np.load(mask_path)
        gt_mask = torch.tensor(gt_mask, dtype=torch.float32).to(device)
        pred_mask = torch.tensor(pred_mask, dtype=torch.float32).to(device)
        dice = dice_score(pred_mask, gt_mask).item()
        dice_scores.append(dice)
    
    return dice_scores

def main():
    # Set device and paths
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/model.pth"
    image_folder = "/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/data/val/images/"
    mask_folder = "/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/data/val/masks/"
    output_folder = "/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/predicted/"

    model = load_model(checkpoint_path, device)
    dice_scores = predict_and_evaluate(model, image_folder, mask_folder, output_folder, device)

    print('Average dice score is', statistics.mean(dice_scores))

if __name__ == "__main__":
    main()