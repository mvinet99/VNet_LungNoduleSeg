import torch
import torch.nn as nn
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-6):       
        
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

SUPPORTED_LOSSES = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "DiceLoss": DiceLoss,
    "DiameterLoss": DiameterLoss
    # Add other loss mappings here if needed
}

class CombinedLoss(nn.Module):
    """Combines multiple weighted losses for segmentation.

    Expects model output logits of shape [B, 1, H, W].
    Expects target labels of shape [B, 1, H, W] or [B, H, W], float type.
    Applies sigmoid internally only for probability-based losses (e.g., Dice).
    """
    def __init__(self, loss_config: Dict[str, Dict], device: torch.device):
        super().__init__()
        self.device = device
        self.losses = nn.ModuleDict() # Use ModuleDict to register losses correctly
        self.weights = {}
        self.requires_sigmoid = set() # Track losses needing sigmoid
        self.requires_diameter = set() # Track losses needing diameter
        total_weight = 0
        active_losses_count = 0

        if not loss_config:
             logger.warning("Empty loss configuration provided. Defaulting to BCEWithLogitsLoss with weight 1.0")
             loss_config = {"BCE": {"name": "BCEWithLogitsLoss", "weight": 1.0, "params": {}}}

        for loss_key, config in loss_config.items(): # e.g., loss_key='BCE', config={name:..., weight:..., params:...}
            loss_name = config.get("name")
            weight = config.get("weight")
            params = config.get("params", {})

            if loss_name in SUPPORTED_LOSSES:
                try:
                    loss_instance = SUPPORTED_LOSSES[loss_name](**params)
                    self.losses[loss_key] = loss_instance.to(self.device)
                    self.weights[loss_key] = weight
                    if loss_name == "DiceLoss": # Add others needing sigmoid here
                        self.requires_sigmoid.add(loss_key)
                    if loss_name == "DiameterLoss":
                        self.requires_diameter.add(loss_key)
                    logger.info(f"Initialized loss component '{loss_key}' ({loss_name}) with params: {params}")
                    active_losses_count += 1
                except Exception as e:
                    logger.error(f"Error initializing loss '{loss_name}' with params {params}: {e}", exc_info=True)
                    raise # Re-raise error to stop execution
            else:
                logger.warning(f"Unsupported loss name '{loss_name}' in config. Skipping.")

        if active_losses_count == 0:
             raise ValueError("No valid loss components were configured.")

        # --- Weight handling/normalization ---
        specified_weights = {k: w for k, w in self.weights.items() if w is not None}

        if len(specified_weights) < active_losses_count:
             # If some weights are None but others are specified, or all are None
             logger.warning(f"Weights not specified for all active losses. Setting equal weights summing to 1.")
             equal_weight = 1.0 / active_losses_count
             for key in self.losses.keys(): # Iterate over *instantiated* losses
                 self.weights[key] = equal_weight
        else:
             # All weights were specified
             total_weight = sum(specified_weights.values())
             if abs(total_weight - 1.0) > 1e-6:
                 logger.warning(f"Loss weights {self.weights} do not sum to 1. Normalizing weights.")
                 # Normalize
                 for key in self.weights:
                     self.weights[key] /= total_weight

        logger.info(f"Final combined loss weights: {self.weights}")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure targets are float and have channel dim
        if targets.dim() == 3: # [B, H, W]
            targets = targets.unsqueeze(1) # [B, 1, H, W]
        targets = targets.float().to(self.device)
        
        # Ensure targets match output shape (B, 1, H, W)
        if targets.shape != outputs.shape:
             logger.error(f"Shape mismatch in CombinedLoss: Output {outputs.shape}, Target {targets.shape}")
             raise ValueError(f"Shape mismatch: Output {outputs.shape}, Target {targets.shape}")
             
        total_loss = torch.tensor(0.0, device=self.device)
        
        predicted_probs = torch.sigmoid(outputs)

        for key, loss_module in self.losses.items():
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                 if key in self.requires_sigmoid:
                     loss_val = loss_module(predicted_probs, targets)
                 else: # Assumes loss works directly on logits
                     loss_val = loss_module(outputs, targets)
                 total_loss += weight * loss_val
        
        return total_loss