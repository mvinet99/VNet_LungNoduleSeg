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

class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It modifies standard BCE loss to focus learning on hard examples and prevent
    the vast number of easy negatives from dominating the loss.

    Args:
        alpha(float): Weighting factor for the rare class (e.g. foreground). Default: 0.25
        gamma(float): Focusing parameter. Higher values down-weight easy examples more. Default: 2.0
        reduction(string): 'mean', 'sum' or 'none'. Default: 'mean'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: raw logits [B, 1, H, W]
        # targets: ground truth [B, 1, H, W]
        
        # Ensure targets are float
        targets = targets.float()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss component
        # Use clamp to prevent log(0) issues
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt (probability of the true class)
        pt = torch.exp(-bce_loss) # pt = p if targets=1; 1-p if targets=0

        # Calculate Focal Loss: alpha * (1 - pt)^gamma * bce_loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

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
    "DiameterLoss": DiameterLoss,
    "FocalLoss": FocalLoss
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
                loss_params = params.copy() # Take a copy to potentially modify
                
                # Handle BCEWithLogitsLoss pos_weight: convert YAML scalar to torch tensor
                if loss_name == "BCEWithLogitsLoss" and 'pos_weight' in loss_params:
                    # Ensure pos_weight is a Python float
                    pos_weight_val = float(loss_params.pop('pos_weight'))
                    # Create tensor on the correct device and dtype
                    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
                    loss_params['pos_weight'] = pos_weight_tensor
                    logger.info(f"Using pos_weight={pos_weight_val} for BCEWithLogitsLoss.")
                
                try:
                    # Instantiate loss with potentially modified params
                    loss_instance = SUPPORTED_LOSSES[loss_name](**loss_params)
                    self.losses[loss_key] = loss_instance.to(self.device)
                    self.weights[loss_key] = weight
                    
                    # Track requirements
                    # DiceLoss and DiameterLoss need sigmoid applied externally
                    if loss_name == "DiceLoss" or loss_name == "DiameterLoss": 
                        self.requires_sigmoid.add(loss_key)
                    # BCEWithLogitsLoss and FocalLoss work on logits directly
                    # No need to track requires_logits explicitly unless needed elsewhere
                    
                    # Removed diameter tracking here as it's handled by requires_sigmoid now
                    # if loss_name == "DiameterLoss":
                    #     self.requires_diameter.add(loss_key)
                        
                    logger.info(f"Initialized loss component '{loss_key}' ({loss_name}) with effective params: {loss_params}")
                    active_losses_count += 1
                except Exception as e:
                    logger.error(f"Error initializing loss '{loss_name}' with effective params {loss_params}: {e}", exc_info=True)
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
        raw_losses = {} # Dictionary to store raw component losses
        
        # Calculate probabilities once if any loss requires them
        predicted_probs = None
        if self.requires_sigmoid:
            predicted_probs = torch.sigmoid(outputs)

        for key, loss_module in self.losses.items():
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                 if key in self.requires_sigmoid:
                     # Use probabilities for losses like Dice
                     if predicted_probs is None: # Should not happen if requires_sigmoid is populated
                         predicted_probs = torch.sigmoid(outputs)
                         logger.warning("Calculating sigmoid probabilities inside loop - inefficient.")
                     loss_val = loss_module(predicted_probs, targets)
                 elif key in self.requires_diameter:
                     # Diameter loss needs probabilities too, but is handled differently
                     if predicted_probs is None:
                         predicted_probs = torch.sigmoid(outputs)
                         logger.warning("Calculating sigmoid probabilities inside loop for diameter - inefficient.")
                     loss_val = loss_module(predicted_probs, targets) # Assuming DiameterLoss takes probs, targets
                 else: # Assumes loss works directly on logits (e.g., BCEWithLogitsLoss, FocalLoss)
                     loss_val = loss_module(outputs, targets)
                 
                 # Store raw (unweighted) loss
                 raw_losses[key] = loss_val.item() # Store as float to prevent holding onto graph
                 
                 # Accumulate weighted loss
                 total_loss += weight * loss_val
        
        return total_loss, raw_losses # Return total loss and dict of raw losses