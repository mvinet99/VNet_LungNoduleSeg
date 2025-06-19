import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Union, Tuple, Dict
import logging
import os
from tqdm import tqdm
from datetime import datetime

class Trainer:
    """
    Trainer class for training a model.

    Args:
        model (nn.Module): The model to train.
        train_loader (data.DataLoader): DataLoader for the training dataset.
        val_loader (data.DataLoader): DataLoader for the validation dataset.
        test_loader (data.DataLoader): DataLoader for the test dataset.
        optimizer (optim.Optimizer): Optimizer for the model.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler for the optimizer.
        criterion (nn.Module): Criterion for the model.
        config (dict): Configuration for the trainer.
        device (torch.device): Device to train on (e.g., 'cuda').
        start_time (str): Unique identifier for the training run.
    """

    def __init__(self, 
                 model: nn.Module, 
                 train_loader: Optional[data.DataLoader]=None, 
                 val_loader: Optional[data.DataLoader]=None,
                 test_loader: Optional[data.DataLoader]=None, 
                 optimizer: Optional[optim.Optimizer]=None, 
                 scheduler: Optional[optim.lr_scheduler._LRScheduler]=None,
                 criterion: Optional[nn.Module] = None, 
                 config: Optional[dict] = None,
                 device: Optional[torch.device]=None,
                 start_time: Optional[str]=None):
        
        self.start_time = start_time if start_time else datetime.now().strftime("%y-%m-%d_%H:%M:%S")
        self.epoch = 1
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Check if optimizer is provided
        if self.optimizer:
            self.logger.info(f"Trainer initialized with optimizer: {type(self.optimizer).__name__}")
        else:
            self.logger.warning("Trainer initialized without an optimizer. Optimizer will be zero.")

        # Check if scheduler is provided
        if self.scheduler:
            self.logger.info(f"Trainer initialized with scheduler: {type(self.scheduler).__name__}")
        else:
            self.logger.info("Trainer initialized without a scheduler.")
        self.criterion = criterion

        # If criterion is provided, move it to the device
        if self.criterion:
            self.criterion.to(self.device)
            self.logger.info(f"Trainer initialized with criterion: {type(self.criterion).__name__}")
        else:
            self.logger.warning("Trainer initialized without a criterion. Loss will be zero.")

        # If no config is provided, create a default one
        self.config = config
        if not self.config:
            configuration = {
                'model': self.model,
                'train_loader': self.train_loader,
                'val_loader': self.val_loader,
                'test_loader': self.test_loader,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'criterion': self.criterion,
                'device': self.device
            }
            self.config = configuration

        # Losses
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        # Dice score
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.test_dice_scores = []

        # Raw Component Losses (NEW)
        self.train_raw_losses = {} # Dict to hold lists like {'BCE': [], 'Dice': []}
        self.val_raw_losses = {}   # Dict to hold lists like {'BCE': [], 'Dice': []}
        self.test_raw_losses = {}  # Dict to hold lists like {'BCE': [], 'Dice': []}

        self.best_val_loss = float('inf')
        self.best_dice_score = 0.0
        self.best_checkpoint_path = None # Path to the best checkpoint saved so far

        # Early Stopping parameters
        self.early_stopping_patience = config.get('training', {}).get('early_stopping_patience', float('inf')) # Default to infinity (no early stopping)
        self.min_delta = config.get('training', {}).get('early_stopping_min_delta', 0.0)
        self.epochs_no_improve = 0
        self.early_stop = False
        if self.early_stopping_patience != float('inf'):
             self.logger.info(f"Early stopping enabled with patience: {self.early_stopping_patience}, min_delta: {self.min_delta}")
        else:
            self.logger.info("Early stopping disabled.")

        # Initialize raw loss lists based on criterion config (NEW)
        if self.criterion and hasattr(self.criterion, 'losses'):
            for loss_key in self.criterion.losses.keys():
                self.train_raw_losses[loss_key] = []
                self.val_raw_losses[loss_key] = []
                self.test_raw_losses[loss_key] = []
            self.logger.debug(f"Initialized raw loss tracking for components: {list(self.criterion.losses.keys())}")
        else:
             self.logger.warning("Could not determine loss components from criterion for raw loss tracking.")

        # Diameter loss
        self.train_diameter_losses = []
        self.val_diameter_losses = []
        self.test_diameter_losses = []
    
    def train(self, num_epochs, save_dir):
        self.model.to(self.device)
        tqdm_epochs = tqdm(range(num_epochs), desc="Epochs")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in tqdm_epochs:
            self.epoch = epoch+1
            train_loss, train_dice, train_diam_loss, train_raw_avg_losses = self.train_one_epoch()
            val_loss, val_dice, val_diam_loss, val_raw_avg_losses = self.validate()
            
            # --- Check for validation DICE improvement for Early Stopping --- 
            # Check before updating best_dice_score
            if val_dice > self.best_dice_score + self.min_delta:
                self.epochs_no_improve = 0 
            else:
                self.epochs_no_improve += 1
                self.logger.debug(f"Epoch {self.epoch}: No improvement in val_dice for {self.epochs_no_improve} epoch(s). Patience: {self.early_stopping_patience}")
            # -------------------------------------------------------------
            
            # --- Step the scheduler --- 
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Pass validation DICE score and ensure scheduler mode is 'max' in config
                    self.scheduler.step(val_dice) 
                    self.logger.debug(f"Stepped ReduceLROnPlateau scheduler with val_dice: {val_dice:.4f}")
                else:
                    self.scheduler.step() # For other schedulers like StepLR, CosineAnnealingLR
                    self.logger.debug(f"Stepped {type(self.scheduler).__name__} scheduler")
                # Log the current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.debug(f"Epoch {self.epoch}: Current learning rate: {current_lr:.2e}")
            # ------------------------

            self.save_checkpoint(save_dir) # Save checkpoint (now based on best Dice)
            
            # --- Log Epoch Metrics at DEBUG level --- 
            self.logger.debug(f"Epoch {self.epoch} Summary:")
            self.logger.debug(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            # Log raw train losses (NEW)
            train_raw_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_raw_avg_losses.items()])
            self.logger.debug(f"  Raw Train Losses: {{{train_raw_str}}}")
            
            self.logger.debug(f"    Val Loss: {val_loss:.4f},   Val Dice: {val_dice:.4f}")
            # Log raw val losses (NEW)
            val_raw_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_raw_avg_losses.items()])
            self.logger.debug(f"  Raw Val Losses: {{{val_raw_str}}}")
            
            self.logger.debug(f"Best Val Loss So Far: {self.best_val_loss:.4f}, Best Val Dice So Far: {self.best_dice_score:.4f}")
            # -------------------------------------

            # --- Update Best Dice Score (after early stopping check) --- 
            if val_dice > self.best_dice_score: # Note: Strict improvement check here
                self.best_dice_score = val_dice
                self.logger.debug(f"Epoch {self.epoch}: New best validation Dice score updated to: {self.best_dice_score:.4f}")
            # --- Update Best Val Loss (for logging) --- 
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # No need to log this update separately unless desired
            # -----------------------------------------
            
            # --- Trigger Early Stopping --- 
            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stop = True
                self.logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement.")
                break # Exit the training loop
            # -------------------------------------
            postfix_dict = {
                'TrainLoss': f"{train_loss:.4f}", 
                'TrainDice': f"{train_dice:.4f}",
                'ValLoss': f"{val_loss:.4f}", 
                'ValDice': f"{val_dice:.4f}",
                'BestValLoss': f"{self.best_val_loss:.4f}",
                'BestValDice': f"{self.best_dice_score:.4f}"
            }
            # Add raw losses to tqdm postfix dynamically (NEW)
            for k, v in train_raw_avg_losses.items():
                postfix_dict[f'TrRaw_{k}'] = f"{v:.3f}"
            for k, v in val_raw_avg_losses.items():
                postfix_dict[f'ValRaw_{k}'] = f"{v:.3f}"
                
            tqdm_epochs.set_postfix(postfix_dict)
        final_checkpoint_path = f'{save_dir}/final_checkpoint_{self.start_time}.pth'
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        final_loss = self.train_losses[-1] if self.train_losses else float('inf')
        final_dice = self.val_dice_scores[-1] if self.val_dice_scores else 0.0
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'loss': final_loss, 
            'dice_score': final_dice, 
            'config': self.config,
            'start_time': self.start_time
        }, final_checkpoint_path)
        self.logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_dice_score = 0.0
        # Initialize accumulators for raw losses (NEW)
        running_raw_losses = {key: 0.0 for key in self.train_raw_losses.keys()}
        
        batch_progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]", leave=False, unit="batch")
        
        for i, (images, labels) in enumerate(batch_progress):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images) # Shape [B, 1, H, W]
            
            if self.criterion:
                # Expect total loss and raw loss dict (NEW)
                loss, raw_losses_batch = self.criterion(outputs, labels) 
            else:
                loss = torch.tensor(0.0, device=self.device)
                raw_losses_batch = {} # Empty dict if no criterion

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_dice = self.dice_score(outputs, labels.float())
            total_dice_score += batch_dice
            
            # Accumulate raw losses (NEW)
            for key, value in raw_losses_batch.items():
                if key in running_raw_losses:
                    running_raw_losses[key] += value # value is already .item() from CombinedLoss
                else:
                     # Should not happen if initialized correctly, but handle defensively
                     self.logger.warning(f"Encountered unexpected raw loss key '{key}' during training.")

        num_batches = len(self.train_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        # Calculate average raw losses (NEW)
        avg_raw_losses = {key: val / num_batches for key, val in running_raw_losses.items()}

        self.train_losses.append(avg_loss)
        self.train_dice_scores.append(avg_dice)
        # Store average raw losses (NEW)
        for key, avg_val in avg_raw_losses.items():
            if key in self.train_raw_losses:
                self.train_raw_losses[key].append(avg_val)

        avg_diameter_loss = 0.0 # Placeholder
        self.train_diameter_losses.append(avg_diameter_loss)
        return avg_loss, avg_dice, avg_diameter_loss, avg_raw_losses # Return avg raw losses
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_dice_score = 0.0
        # Initialize accumulators for raw losses (NEW)
        running_raw_losses = {key: 0.0 for key in self.val_raw_losses.keys()}
        
        batch_progress = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for i, (images, labels) in enumerate(batch_progress):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images) # Shape [B, 1, H, W]
                
                if self.criterion:
                    # Expect total loss and raw loss dict (NEW)
                    loss, raw_losses_batch = self.criterion(outputs, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                    raw_losses_batch = {}
                running_loss += loss.item()
                
                total_dice_score += self.dice_score(outputs, labels.float())
                
                # Accumulate raw losses (NEW)
                for key, value in raw_losses_batch.items():
                     if key in running_raw_losses:
                         running_raw_losses[key] += value # value is already .item() from CombinedLoss
                     else:
                         self.logger.warning(f"Encountered unexpected raw loss key '{key}' during validation.")

        num_batches = len(self.val_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        # Calculate average raw losses (NEW)
        avg_raw_losses = {key: val / num_batches for key, val in running_raw_losses.items()}
        
        self.val_losses.append(avg_loss)
        self.val_dice_scores.append(avg_dice)
        # Store average raw losses (NEW)
        for key, avg_val in avg_raw_losses.items():
            if key in self.val_raw_losses:
                 self.val_raw_losses[key].append(avg_val)

        avg_diameter_loss = 0.0 # Placeholder
        self.val_diameter_losses.append(avg_diameter_loss)
        return avg_loss, avg_dice, avg_diameter_loss, avg_raw_losses # Return avg raw losses

    def test(self):
        self.model.eval()

        running_loss = 0.0
        total_dice_score = 0.0
        total_diameter_loss = 0.0
        # Initialize accumulators for raw losses (NEW)
        running_raw_losses = {key: 0.0 for key in self.test_raw_losses.keys()}

        batch_progress = tqdm(self.test_loader, desc=f"[Test]", leave=False, unit="batch")

        with torch.no_grad():
            for i, (images, labels) in enumerate(batch_progress):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                # Expect total loss and raw loss dict (NEW)
                loss, raw_losses_batch = self.criterion(outputs, labels) if self.criterion else (torch.tensor(0.0, device=self.device), {})

                running_loss += loss.item()
                total_dice_score += self.dice_score(outputs, labels.float())
                
                # Accumulate raw losses (NEW)
                for key, value in raw_losses_batch.items():
                     if key in running_raw_losses:
                         running_raw_losses[key] += value
                     else:
                          self.logger.warning(f"Encountered unexpected raw loss key '{key}' during testing.")

        num_batches = len(self.test_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        avg_diameter_loss = total_diameter_loss / num_batches
        # Calculate average raw losses (NEW)
        avg_raw_losses = {key: val / num_batches for key, val in running_raw_losses.items()}

        self.test_losses.append(avg_loss)
        self.test_dice_scores.append(avg_dice)
        self.test_diameter_losses.append(avg_diameter_loss)
        # Store average raw losses (NEW)
        for key, avg_val in avg_raw_losses.items():
            if key in self.test_raw_losses:
                 self.test_raw_losses[key].append(avg_val)

        self.logger.info(f"--- Testing Complete ---")
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        self.logger.info(f"Test Dice Score: {avg_dice:.4f}")
        # Log raw test losses (NEW)
        test_raw_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_raw_losses.items()])
        self.logger.info(f"Raw Test Losses: {{{test_raw_str}}}")

    def save_checkpoint(self, save_dir):
        """Saves the model checkpoint if the current validation Dice score is the best so far."""
        # Ensure we have validation Dice scores available
        if not self.val_dice_scores:
            self.logger.warning("Validation Dice scores not available yet, cannot save best checkpoint.")
            return

        current_val_dice = self.val_dice_scores[-1]
        # Save checkpoint based on best validation Dice
        if current_val_dice > self.best_dice_score:
            previous_best_path = self.best_checkpoint_path
            new_best_checkpoint_path = f"{save_dir}/best_checkpoint_{self.start_time}_epoch_{self.epoch}.pth"
            optimizer_state = self.optimizer.state_dict() if self.optimizer else None
            current_train_loss = self.train_losses[-1] if self.train_losses else float('inf')
            current_val_loss = self.val_losses[-1] if self.val_losses else None

            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'loss_train': current_train_loss,
                'loss_val': current_val_loss,
                'dice_score_val': current_val_dice,
                'config': self.config,
                'start_time': self.start_time
            }
            os.makedirs(os.path.dirname(new_best_checkpoint_path), exist_ok=True)
            torch.save(checkpoint, new_best_checkpoint_path)
            self.logger.debug(f"Saved new best checkpoint to {new_best_checkpoint_path} with val Dice: {current_val_dice:.4f}")

            # Update best Dice and checkpoint path
            self.best_dice_score = current_val_dice
            self.best_checkpoint_path = new_best_checkpoint_path

            # Remove the previous best checkpoint file if it exists
            if previous_best_path and os.path.exists(previous_best_path):
                try:
                    os.remove(previous_best_path)
                    self.logger.debug(f"Removed previous best checkpoint: {previous_best_path}")
                except OSError as e:
                    self.logger.error(f"Error removing previous best checkpoint {previous_best_path}: {e}")
        else:
            self.logger.debug(f"Epoch {self.epoch}: Val Dice {current_val_dice:.4f} did not improve over best {self.best_dice_score:.4f}")

    def _log_epoch_metrics(self):
        self.logger.info(f"--- Epoch {self.epoch} Summary ---")
        if self.train_losses:
            self.logger.info(f"Train Loss: {self.train_losses[-1]:.4f}")
        if self.train_dice_scores:
            self.logger.info(f"Train Dice Score: {self.train_dice_scores[-1]:.4f}")
        if self.train_diameter_losses:
            self.logger.info(f"Train Diameter Loss: {self.train_diameter_losses[-1]:.4f}")

        if self.val_losses:
            self.logger.info(f"Val Loss: {self.val_losses[-1]:.4f}")
        if self.val_dice_scores:
            self.logger.info(f"Val Dice Score: {self.val_dice_scores[-1]:.4f}")
        if self.val_diameter_losses:
            self.logger.info(f"Val Diameter Loss: {self.val_diameter_losses[-1]:.4f}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def dice_score(self, outputs, target_labels_float):
        # outputs: raw logits from model, shape [B, 1, H, W]
        # target_labels_float: ground truth mask, shape [B, 1, H, W] or [B, H, W], type float
        
        outputs = outputs.to(self.device)
        target_labels_float = target_labels_float.to(self.device)
        
        # Ensure target has same shape as output (if needed)
        if target_labels_float.shape != outputs.shape:
             if target_labels_float.dim() == 3 and outputs.dim() == 4 and outputs.shape[1] == 1:
                 target_labels_float = target_labels_float.unsqueeze(1)
             elif target_labels_float.shape != outputs.shape: 
                  self.logger.error(f"Dice score shape mismatch: Output {outputs.shape}, Target {target_labels_float.shape}")
                  return 0.0 

        predicted_probs = torch.sigmoid(outputs)
        predicted_mask = (predicted_probs > 0.5).float()
        pred_flat = predicted_mask.contiguous().view(-1)
        target_flat = target_labels_float.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coefficient = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        return dice_coefficient.item() 
