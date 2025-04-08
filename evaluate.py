import os
import torch
import numpy as np
import statistics
import random
from model import VNet

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
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * target)
    return (2. * intersection) / (torch.sum(pred) + torch.sum(target) + 1e-6)

# Load the model
def load_model(checkpoint_path:os.PathLike, device:torch.device):
    model = VNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
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