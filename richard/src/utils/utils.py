import os
import torch
import numpy as np
import statistics
import random
from model import VNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import time
import logging
from pathlib import Path
from ruamel.yaml import YAML
from typing import Optional, Dict

class DiceLoss(nn.Module):
    """Computes Dice Loss for binary segmentation.

    Assumes input is probabilities (e.g., after sigmoid) and target is binary mask.
    Input and target tensors are expected to be flattened or will be flattened.
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        targets = targets.float()
        intersection = (inputs * targets).sum()
        denominator = inputs.sum() + targets.sum()
        dice_coefficient = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1. - dice_coefficient

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

# Model debugging decorator
def setup_logging(level=logging.DEBUG):

    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

def debug_decorator(func):
    """Decorator to conditionally log debug information based on self.debug"""
    # Create a module-level logger - each module gets its own logger
    logger = logging.getLogger(__name__)
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        debug_active = getattr(self, 'debug', False)
        func_name = self.__class__.__name__ + "." + func.__name__
        
        # Log input if debug is active
        if debug_active:
            logger.debug(f"Entering {func_name}")
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    logger.debug(f"Arg {i} shape: {arg.shape}")
        
        # Start a timer for performance monitoring
        start_time = time.time()
        
        # Call the original function
        output = func(self, *args, **kwargs)
        
        # Log output and performance info if debug is active
        if debug_active:
            execution_time = time.time() - start_time
            logger.debug(f"{func_name} executed in {execution_time:.5f} seconds")
            
            if isinstance(output, torch.Tensor):
                logger.debug(f"Output shape: {output.shape}")
                logger.debug(f"Output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
            
            logger.debug(f"Exiting {func_name}")
        
        return output
    return wrapper

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

def set_visible_devices(device_ids=None):
    """Set the visible devices for PyTorch.

    This function sets the visible devices for PyTorch to ensure that the same
    GPU is used for each run. If no device IDs are provided, it will use all
    available devices
    """

    if device_ids:
        if isinstance(device_ids, str):
            device_ids = [int(id) for id in device_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    else:
        # Show all devices (gpustat)
        os.system("gpustat")
        # Get user input for device IDs
        user_input = str(input("Enter the device IDs to use (example '0,1') : "))
        device_ids = [int(id) for id in user_input.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
# --- Configuration Loading Helper ---
def _load_and_merge_yaml_configs(main_config_path: Path) -> dict:
    yaml = YAML()
    config_dir = main_config_path.parent
    merged_cfg = {}
    logger = logging.getLogger(__name__) # Use logger defined in this module

    try:
        with open(main_config_path, 'r') as f:
            base_cfg = yaml.load(f)
            logger.info(f"Decorator loaded base config from: {main_config_path}")
    except FileNotFoundError:
        logger.error(f"Decorator: Main configuration file not found: {main_config_path}")
        raise # Re-raise exception to signal failure
    except Exception as e:
        logger.error(f"Decorator: Error loading main configuration {main_config_path}: {e}")
        raise # Re-raise exception

    for key, value_name in base_cfg.items():
        # Construct path: e.g., richard/config/model/vnet.yaml
        if not isinstance(value_name, dict) and not isinstance(value_name, list):
            specific_config_path = config_dir / key / f"{value_name}.yaml"
            try:
                with open(specific_config_path, 'r') as f:
                    specific_cfg = yaml.load(f)
                    # Store the loaded dictionary under the key (e.g., merged_cfg['model'] = {content of vnet.yaml})
                    merged_cfg[key] = specific_cfg
                    logger.info(f"Decorator loaded config for '{key}' from: {specific_config_path}")
            except FileNotFoundError:
                logger.error(f"Decorator: Specific configuration file not found for key '{key}': {specific_config_path}")
                # Decide behavior: Assign empty dict or raise error? Raising is often safer.
                # merged_cfg[key] = {}
                raise FileNotFoundError(f"Missing specific config file: {specific_config_path}")
            except Exception as e:
                logger.error(f"Decorator: Error loading specific configuration {specific_config_path}: {e}")
                # merged_cfg[key] = {}
                raise # Re-raise exception
        else:
            merged_cfg[key] = value_name

    return merged_cfg

# --- Configuration Loading Decorator ---
def load_config_decorator(config_arg_name: str = "config"):
    """
    Decorator to load and merge YAML configurations specified in a base file.

    Assumes the decorated function receives an object (like argparse.Namespace)
    as its first argument (`args`), and that object has an attribute named
    `config_arg_name` containing the path to the main config file.

    Injects the loaded config dictionary as a keyword argument 'cfg' into the
    decorated function call.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__) # Use logger
            # --- Find the config path ---
            main_config_path_str = None
            args_obj = None
            if args:
                # Assuming the first positional argument holds the config path attribute
                args_obj = args[0]
                if hasattr(args_obj, config_arg_name):
                    main_config_path_str = getattr(args_obj, config_arg_name)
                else:
                    logger.error(f"Decorator: Argument object missing attribute '{config_arg_name}'.")
                    raise AttributeError(f"Argument object missing attribute '{config_arg_name}'.")
            else:
                logger.error("Decorator: No positional arguments found to retrieve config path from.")
                raise ValueError("Decorator requires positional arguments to find config path.")


            if main_config_path_str is None:
                 # This case should ideally be caught by checks above
                logger.error(f"Decorator could not find config path string for arg '{config_arg_name}'.")
                raise ValueError(f"Config path attribute '{config_arg_name}' did not provide a path.")

            main_config_path = Path(main_config_path_str)

            # --- Load and merge config ---
            try:
                loaded_cfg = _load_and_merge_yaml_configs(main_config_path)
                logger.info("Decorator successfully loaded and merged configurations.")
            except Exception as e:
                 logger.error(f"Decorator failed during config loading: {e}", exc_info=True)
                 # Stop execution if config loading fails
                 raise # Re-raise the exception encountered during loading

            # --- Call original function with injected config ---
            # Pass original args/kwargs along, and add 'cfg'
            logger.info("Decorator injecting loaded 'cfg' into function call.")
            # Check if 'cfg' is already in kwargs to prevent potential conflicts
            if 'cfg' in kwargs:
                 logger.warning(f"Decorator attempting to inject 'cfg', but it already exists in kwargs. Overwriting.")

            # Ensure we don't modify the original kwargs dict if not necessary,
            # but create a new one if 'cfg' needs to be added.
            new_kwargs = kwargs.copy()
            new_kwargs['cfg'] = loaded_cfg

            return func(*args, **new_kwargs) # Inject cfg via updated kwargs

        return wrapper
    return decorator

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