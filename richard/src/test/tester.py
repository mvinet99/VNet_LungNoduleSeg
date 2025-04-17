import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt # For visualization

logger = logging.getLogger(__name__)

class Tester:
    """Handles the evaluation and visualization of a model on a test dataset."""

    def __init__(self, 
                 model: nn.Module, 
                 test_loader: data.DataLoader,
                 criterion: nn.Module, 
                 device: torch.device = torch.device("cuda"),
                 output_dir: Path = Path("/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results"),
                 thresh_min: float = 0.0,
                 thresh_max: float = 1.0,
                 thresh_step: float = 0.05,
                 num_visual_samples: int = 10,
                 overlay_opacity: float = 0.4):
        """
        Args:
            model: The trained model to evaluate.
            test_loader: DataLoader for the test dataset.
            criterion: Loss function (used for calculating test loss).
            device: Device to run inference on (e.g., 'cuda').
            output_dir: Directory to save predicted masks and overlay images.
            thresholds: A list of thresholds to evaluate the Dice score at.
            num_visual_samples: Number of samples to generate overlay visualizations for.
            overlay_opacity: Opacity level for the mask overlays (0.0 to 1.0).
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion.to(device) if criterion else None
        self.device = device
        self.output_dir = Path(output_dir)
        self.thresholds = sorted(np.arange(thresh_min, thresh_max + thresh_step, thresh_step))
        self.num_visual_samples = num_visual_samples
        self.overlay_opacity = overlay_opacity
        
        self.pred_mask_dir = self.output_dir / "predicted_masks"
        self.overlay_dir = self.output_dir / "overlays"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_mask_dir.mkdir(exist_ok=True)
        self.overlay_dir.mkdir(exist_ok=True)

        logger.info(f"Tester initialized. Results will be saved to: {self.output_dir}")
        logger.info(f"Evaluating Dice at thresholds: {self.thresholds}")
        logger.info(f"Generating {self.num_visual_samples} visualization samples.")

    def _calculate_dice_score(self, predicted_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculates the Dice score between a predicted and target mask."""
        pred_flat = predicted_mask.contiguous().view(-1)
        target_flat = target_mask.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coefficient = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return dice_coefficient.item()

    def evaluate(self) -> tuple[float, float, dict[float, float]]:
        """Evaluates the model on the test set across multiple thresholds and returns best Dice and threshold."""
        self.model.eval()
        # Initialize dict to store total dice score sum for each threshold
        total_dice_scores = {thresh: 0.0 for thresh in self.thresholds}
        num_samples = 0

        logger.info("Starting evaluation...")
        batch_progress = tqdm(self.test_loader, desc="[Test Eval]", leave=False, unit="batch")

        with torch.no_grad():
            for i, (images, labels) in enumerate(batch_progress):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                # Evaluate Dice for each threshold for this batch
                for thresh in self.thresholds:
                    predicted_mask = (probs > thresh).float() # Shape [B, 1, H, W]
                    # Calculate dice for each sample in the batch and sum
                    batch_dice_sum = 0
                    for single_pred, single_label in zip(predicted_mask, labels):
                        batch_dice_sum += self._calculate_dice_score(single_pred, single_label)
                    total_dice_scores[thresh] += batch_dice_sum # Accumulate sum over all samples

                num_samples += images.size(0) # Count total samples processed

        avg_dice_by_threshold = {thresh: total / num_samples for thresh, total in total_dice_scores.items()}
        # Find best threshold and Dice
        best_threshold = max(avg_dice_by_threshold, key=avg_dice_by_threshold.get)
        best_dice = avg_dice_by_threshold[best_threshold]

        logger.info("--- Evaluation Complete ---")
        logger.info(f"Average Dice Scores per Threshold:")
        for thresh, score in avg_dice_by_threshold.items():
            logger.info(f"  Threshold {thresh:.2f}: {score:.4f}")
        logger.info(f"Best Average Dice Score: {best_dice:.4f} at Threshold: {best_threshold:.2f}")

        return best_dice, best_threshold, avg_dice_by_threshold

    def _create_overlay(self, image: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, filename: Path):
        """Creates and saves an overlay image."""
        # Assuming image is [H, W], masks are [H, W]
        if image.ndim == 3 and image.shape[0] == 1: # Handle [1, H, W]
             image = image.squeeze(0)
        if true_mask.ndim == 3 and true_mask.shape[0] == 1:
             true_mask = true_mask.squeeze(0)
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
             pred_mask = pred_mask.squeeze(0)
        
        # Normalize image to 0-1 for display
        img_normalized = (image - image.min()) / (image.max() - image.min() + 1e-6)
        img_rgb = np.stack([img_normalized] * 3, axis=-1) # Convert to RGB

        # Create colored masks
        true_mask_colored = np.zeros_like(img_rgb)
        true_mask_colored[true_mask > 0.5] = [1, 0, 0] # Red for true mask
        
        pred_mask_colored = np.zeros_like(img_rgb)
        pred_mask_colored[pred_mask > 0.5] = [0, 1, 0] # Green for predicted mask

        # Blend image and masks
        overlay = img_rgb * (1.0 - self.overlay_opacity) # Dim image
        overlay[true_mask > 0.5] = overlay[true_mask > 0.5] + true_mask_colored[true_mask > 0.5] * self.overlay_opacity
        overlay[pred_mask > 0.5] = overlay[pred_mask > 0.5] + pred_mask_colored[pred_mask > 0.5] * self.overlay_opacity
        
        # Clip values to [0, 1]
        overlay = np.clip(overlay, 0, 1)

        try:
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(f"Overlay (True=Red, Pred=Green)")
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close() # Close figure to free memory
        except Exception as e:
            logger.error(f"Failed to save overlay image {filename}: {e}")


    def generate_visualizations(self, best_threshold: float):
        """Generates and saves predicted masks and overlay visualizations."""
        self.model.eval()
        samples_saved = 0
        
        logger.info(f"Generating visualizations using threshold {best_threshold:.2f}...")
        
        # Determine a unique base filename for each sample
        # Using dataset indices might be ideal if the loader provides them.
        # As a fallback, use enumeration index.
        
        # Iterate through loader to get samples
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if samples_saved >= self.num_visual_samples:
                    break # Stop once enough samples are saved

                images = images.to(self.device)
                labels = labels.to(self.device).float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                predicted_masks = (probs > best_threshold).float()

                # Save each sample in the batch until limit is reached
                for j in range(images.size(0)):
                    if samples_saved >= self.num_visual_samples:
                        break

                    # Define filenames based on batch index i and sample index j
                    # If dataset provides filenames/ids, use those instead.
                    base_filename = f"sample_{i*self.test_loader.batch_size + j:04d}" 
                    pred_mask_path = self.pred_mask_dir / f"{base_filename}_predmask.npy"
                    overlay_path = self.overlay_dir / f"{base_filename}_overlay.png"

                    # Get single sample data (move to CPU, convert to NumPy)
                    img_np = images[j].cpu().numpy().squeeze() # Assuming single channel input
                    label_np = labels[j].cpu().numpy().squeeze()
                    pred_mask_np = predicted_masks[j].cpu().numpy().squeeze()

                    # Save predicted mask as numpy array
                    try:
                        np.save(pred_mask_path, pred_mask_np)
                    except Exception as e:
                         logger.error(f"Failed to save predicted mask {pred_mask_path}: {e}")

                    # Create and save overlay
                    self._create_overlay(img_np, label_np, pred_mask_np, overlay_path)
                    
                    samples_saved += 1
        
        logger.info(f"Saved {samples_saved} visualization samples to {self.overlay_dir} and masks to {self.pred_mask_dir}") 