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
                 num_visual_patients: int = 10,
                 overlay_opacity: float = 0.4,
                 save_masks: bool = False,
                 save_overlays: bool = False):
        """
        Args:
            model: The trained model to evaluate.
            test_loader: DataLoader for the test dataset.
            criterion: Loss function (used for calculating test loss).
            device: Device to run inference on (e.g., 'cuda').
            output_dir: Directory to save predicted masks and overlay images.
            thresholds: A list of thresholds to evaluate the Dice score at.
            num_visual_patients: Number of patients to generate overlay visualizations for.
            overlay_opacity: Opacity level for the mask overlays (0.0 to 1.0).
            save_masks (bool): If True, save predicted masks as .npy files.
            save_overlays (bool): If True, save visual overlay images.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion.to(device) if criterion else None
        self.device = device
        self.output_dir = Path(output_dir)
        self.thresholds = sorted(np.arange(thresh_min, thresh_max + thresh_step, thresh_step))
        self.num_visual_patients = num_visual_patients
        self.overlay_opacity = overlay_opacity
        self.save_masks = save_masks
        self.save_overlays = save_overlays
        
        self.pred_mask_dir = self.output_dir / "predicted_masks"
        self.overlay_dir = self.output_dir / "overlays"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_masks:
            self.pred_mask_dir.mkdir(exist_ok=True)
        if self.save_overlays:
            self.overlay_dir.mkdir(exist_ok=True)

        logger.info(f"Tester initialized. Results will be saved to: {self.output_dir}")
        logger.info(f"Save predicted masks: {self.save_masks}")
        logger.info(f"Save overlay visuals: {self.save_overlays}")
        if self.save_overlays:
            logger.info(f"Generating visualizations for {self.num_visual_patients} patients.")
        logger.info(f"Evaluating Dice at thresholds: {self.thresholds}")

    def _extract_patient_id(self, filename: str) -> str:
        """Extract patient ID from filename.
        
        Format examples: NLST994_slice_29.npy, UCLA123_slice_45.npy
        Returns: NLST994 or UCLA123
        """
        # Split by first underscore to get the patient identifier
        parts = filename.split('_', 1)
        return parts[0] if parts else filename
    
    def _calculate_dice_score(self, predicted_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculates the Dice score between a predicted and target mask."""
        pred_flat = predicted_mask.contiguous().view(-1)
        target_flat = target_mask.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coefficient = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return dice_coefficient.item()

    def _get_batch_filenames(self, batch_idx):
        """Helper method to get filenames for the current batch."""
        start_idx = batch_idx * self.test_loader.batch_size
        end_idx = min(start_idx + self.test_loader.batch_size, len(self.test_loader.dataset))
        return [self.test_loader.dataset.images[idx] for idx in range(start_idx, end_idx)]

    def evaluate(self) -> tuple[float, float, dict[float, float]]:
        """Evaluates the model on the test set per patient across multiple thresholds 
        and returns the best average patient Dice score and corresponding threshold.
        """
        self.model.eval()
        patient_probs = {}    # Store {patient_id: [probabilities]}
        patient_labels = {}   # Store {patient_id: [labels]}
        
        # Initialize dictionary to store dice scores per threshold
        patient_dice_by_threshold = {thresh: [] for thresh in self.thresholds}
        
        logger.info("Starting evaluation...")
        batch_progress = tqdm(self.test_loader, desc="[Test Eval]", leave=False, unit="batch")

        # Collect all probabilities and labels per patient
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(batch_progress):
                # Get filenames for current batch
                filenames = self._get_batch_filenames(batch_idx)
                
                # Process images through model
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                
                # Group results by patient ID
                for slice_idx, filename in enumerate(filenames):
                    patient_id = self._extract_patient_id(filename)
                    
                    # Initialize patient entry if first occurrence
                    if patient_id not in patient_probs:
                        patient_probs[patient_id] = []
                        patient_labels[patient_id] = []
                    
                    # Store probability map and label for this slice
                    patient_probs[patient_id].append(probs[slice_idx].cpu())
                    patient_labels[patient_id].append(labels[slice_idx].cpu())
                
                batch_progress.set_postfix({"Patients": len(patient_probs)})
        
        # Calculate Dice score for each patient at each threshold
        logger.info(f"Computing dice scores for {len(patient_probs)} patients...")
        for patient_id, prob_slices in patient_probs.items():
            # Stack all slices for this patient
            patient_prob_volume = torch.stack(prob_slices)      # [num_slices, 1, H, W]
            patient_label_volume = torch.stack(patient_labels[patient_id])
            
            # Calculate Dice score at each threshold
            for thresh in self.thresholds:
                patient_pred_volume = (patient_prob_volume > thresh).float()
                dice = self._calculate_dice_score(patient_pred_volume, patient_label_volume)
                patient_dice_by_threshold[thresh].append(dice)

        # Calculate average Dice score across patients for each threshold
        avg_dice_by_threshold = {
            thresh: sum(scores) / len(scores) if scores else 0.0 
            for thresh, scores in patient_dice_by_threshold.items()
        }
        
        # Find best threshold and Dice
        best_threshold = max(avg_dice_by_threshold, key=avg_dice_by_threshold.get)
        best_dice = avg_dice_by_threshold[best_threshold]

        # Log results
        logger.info("--- Evaluation Complete ---")
        logger.info(f"Processed {len(patient_probs)} patients")
        logger.info(f"Average Patient Dice Scores per Threshold:")
        for thresh, score in sorted(avg_dice_by_threshold.items()):
            logger.info(f"  Threshold {thresh:.2f}: {score:.4f}")
        logger.info(f"Best Average Patient Dice Score: {best_dice:.4f} at Threshold: {best_threshold:.2f}")

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


    def generate_visualizations(self, best_threshold: float, new_only: bool = True):
        """Generates and saves predicted masks and overlay visualizations for a specific number of patients.

        Args:
            best_threshold: The probability threshold used to create binary masks.
            new_only: If True, only generate visualizations for patients who do not
                      already have an existing visualization directory.
        """
        self.model.eval()
        patient_dirs = {}  # Track patient directories: {patient_id: (pred_dir, overlay_dir)}
        processed_patients = set()  # Track which patients we've already processed
        
        # Early exit if no visualizations are requested
        if not self.save_masks and not self.save_overlays:
            logger.info("Skipping visualization generation as both save_masks and save_overlays are False.")
            return

        logger.info(f"Generating visualizations for up to {self.num_visual_patients} patients using threshold {best_threshold:.2f}.")
        if new_only:
            logger.info("Flag 'new_only' is set. Skipping patients with existing visualizations.")

        # First pass: collect all patient IDs from the dataset
        all_patient_ids = set()
        for batch_idx in range(len(self.test_loader)):
            filenames = self._get_batch_filenames(batch_idx)
            for filename in filenames:
                patient_id = self._extract_patient_id(filename)
                all_patient_ids.add(patient_id)
        
        logger.info(f"Found {len(all_patient_ids)} total patients in dataset")
        
        # Filter out patients with existing visualizations if new_only is True
        patients_for_consideration = all_patient_ids
        if new_only:
            existing_patient_ids = set()
            for patient_id in all_patient_ids:
                patient_overlay_dir = self.overlay_dir / patient_id
                if patient_overlay_dir.exists() and any(patient_overlay_dir.iterdir()): # Check if dir exists and is not empty
                    existing_patient_ids.add(patient_id)
            
            if existing_patient_ids:
                logger.info(f"Skipping {len(existing_patient_ids)} patients with existing visualization directories.")
                patients_for_consideration = all_patient_ids - existing_patient_ids
            else:
                logger.info("No existing patient visualization directories found to skip.")

        # Decide which patients to visualize (take first num_visual_patients from the considered list)
        patients_to_visualize = sorted(list(patients_for_consideration))[:self.num_visual_patients]
        
        if patients_to_visualize:
            logger.info(f"Will generate visualizations for patients: {', '.join(patients_to_visualize)}")
        else:
            logger.info("No new patients selected for visualization based on current settings.")
            # No need to proceed further if no patients are selected
            return

        # Second pass: process the data and save visualizations for selected patients
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="[Visualizing]", leave=False, unit="batch")
            for batch_idx, (images, labels) in enumerate(progress_bar):
                # Get filenames for the current batch
                filenames = self._get_batch_filenames(batch_idx)
                
                # Check if any filename in this batch belongs to a patient we want to visualize
                batch_has_target_patient = False
                for filename in filenames:
                    patient_id = self._extract_patient_id(filename)
                    if patient_id in patients_to_visualize:
                        batch_has_target_patient = True
                        break
                
                if not batch_has_target_patient:
                    continue  # Skip this batch if no target patients
                
                # Process images through model
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                predicted_masks = (probs > best_threshold).float()

                # Save slices for targeted patients
                for slice_idx, filename in enumerate(filenames):
                    patient_id = self._extract_patient_id(filename)
                    
                    # Skip if this patient is not in our target list
                    if patient_id not in patients_to_visualize:
                        continue
                        
                    # Create patient-specific directories if needed
                    if patient_id not in patient_dirs:
                        patient_pred_dir = self.pred_mask_dir / patient_id
                        patient_overlay_dir = self.overlay_dir / patient_id
                        if self.save_masks:
                            patient_pred_dir.mkdir(exist_ok=True)
                        if self.save_overlays:
                            patient_overlay_dir.mkdir(exist_ok=True)
                        patient_dirs[patient_id] = (patient_pred_dir, patient_overlay_dir)
                        processed_patients.add(patient_id)
                    
                    # Use the actual filename (without extension) for saving results
                    base_filename = filename.split('.')[0]  # Remove file extension
                    pred_mask_path = patient_dirs[patient_id][0] / f"{base_filename}_predmask.npy"
                    overlay_path = patient_dirs[patient_id][1] / f"{base_filename}_overlay.png"

                    # Extract data for current slice
                    img_np = images[slice_idx].cpu().numpy().squeeze()
                    label_np = labels[slice_idx].cpu().numpy().squeeze()
                    pred_mask_np = predicted_masks[slice_idx].cpu().numpy().squeeze()

                    # Save predicted mask
                    if self.save_masks:
                        try:
                            np.save(pred_mask_path, pred_mask_np)
                        except Exception as e:
                            logger.error(f"Failed to save predicted mask {pred_mask_path}: {e}")

                    # Create and save overlay
                    if self.save_overlays:
                        self._create_overlay(img_np, label_np, pred_mask_np, overlay_path)
                
                # Early exit if we've processed all target patients
                if len(processed_patients) >= self.num_visual_patients:
                    break
        
        # Count total slices saved
        total_slices = 0
        for patient_id in processed_patients:
            pred_dir = patient_dirs[patient_id][0]
            patient_slices = len(list(pred_dir.glob("*_predmask.npy")))
            total_slices += patient_slices
            logger.info(f"Patient {patient_id}: saved {patient_slices} slices")
        
        logger.info(f"Saved visualizations for {len(processed_patients)} patients ({total_slices} total slices)") 