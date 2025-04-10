import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from typing import Optional, Union, Tuple, Dict
import logging
import os
from tqdm import tqdm
from richard.src.utils.utils import DiceLoss

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: data.DataLoader, 
                 val_loader: data.DataLoader, 
                 test_loader: Optional[data.DataLoader]=None, 
                 optimizer: Optional[optim.Optimizer]=None, 
                 criterion: Optional[nn.Module] = None, 
                 device: Optional[torch.device]=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.criterion = criterion
        if self.criterion:
            self.criterion.to(self.device)
            self.logger.info(f"Trainer initialized with criterion: {type(self.criterion).__name__}")
        else:
            self.logger.warning("Trainer initialized without a criterion. Loss will be zero.")

        # Losses
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        # Dice score
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.test_dice_scores = []
        self.best_dice_score = 0.0
        self.best_dice_score_epoch = 0
        self.best_dice_score_model_path = None

        # Diameter loss
        self.train_diameter_losses = []
        self.val_diameter_losses = []
        self.test_diameter_losses = []
    
    def train(self, num_epochs, save_dir):
        self.model.to(self.device)
        tqdm_epochs = tqdm(range(num_epochs), desc="Epochs")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in tqdm_epochs:
            self.epoch = epoch
            train_loss, train_dice, train_diam_loss = self.train_one_epoch()
            val_loss, val_dice, val_diam_loss = self.validate()
            self.update_best_dice_score(save_dir)
            self.save_checkpoint(save_dir)
            postfix_dict = {
                'TrainLoss': f"{train_loss:.4f}", 
                'TrainDice': f"{train_dice:.4f}",
                'ValLoss': f"{val_loss:.4f}", 
                'ValDice': f"{val_dice:.4f}",
                'BestDice': f"{self.best_dice_score:.4f}"
            }
            tqdm_epochs.set_postfix(postfix_dict)
        final_checkpoint_path = f'{save_dir}/final_checkpoint.pth'
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        final_loss = self.train_losses[-1] if self.train_losses else float('inf')
        final_dice = self.val_dice_scores[-1] if self.val_dice_scores else 0.0
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'loss': final_loss, 
            'dice_score': final_dice, 
            'model_path': final_checkpoint_path
        }, final_checkpoint_path)
        self.logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_dice_score = 0.0
        batch_progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]", leave=False, unit="batch")
        
        for i, (images, labels) in enumerate(batch_progress):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images) # Shape [B, 1, H, W]
            
            if self.criterion:
                loss = self.criterion(outputs, labels)
            else:
                loss = torch.tensor(0.0, device=self.device)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_dice = self.dice_score(outputs, labels.float())
            total_dice_score += batch_dice

        num_batches = len(self.train_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        self.train_losses.append(avg_loss)
        self.train_dice_scores.append(avg_dice)
        avg_diameter_loss = 0.0 # Placeholder
        self.train_diameter_losses.append(avg_diameter_loss)
        return avg_loss, avg_dice, avg_diameter_loss
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_dice_score = 0.0
        batch_progress = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for i, (images, labels) in enumerate(batch_progress):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images) # Shape [B, 1, H, W]
                
                if self.criterion:
                    loss = self.criterion(outputs, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                running_loss += loss.item()
                
                total_dice_score += self.dice_score(outputs, labels.float())

        num_batches = len(self.val_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        self.val_losses.append(avg_loss)
        self.val_dice_scores.append(avg_dice)
        avg_diameter_loss = 0.0 # Placeholder
        self.val_diameter_losses.append(avg_diameter_loss)
        return avg_loss, avg_dice, avg_diameter_loss

    def test(self):
        self.model.eval()

        running_loss = 0.0
        total_dice_score = 0.0
        total_diameter_loss = 0.0

        batch_progress = tqdm(self.test_loader, desc=f"Epoch {self.epoch+1} [Test]", leave=False, unit="batch")

        with torch.no_grad():
            for i, (images, labels) in enumerate(batch_progress):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels) if self.criterion else torch.tensor(0.0, device=self.device)

                running_loss += loss.item()
                total_dice_score += self.dice_score(outputs, labels.float())

        num_batches = len(self.test_loader)
        avg_loss = running_loss / num_batches
        avg_dice = total_dice_score / num_batches
        avg_diameter_loss = total_diameter_loss / num_batches

        self.test_losses.append(avg_loss)
        self.test_dice_scores.append(avg_dice)
        self.test_diameter_losses.append(avg_diameter_loss)

        self.logger.info(f"--- Testing Complete ---")
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        self.logger.info(f"Test Dice Score: {avg_dice:.4f}")

    def save_checkpoint(self, save_dir):
        if not self.val_dice_scores:
            self.logger.warning("Validation scores not available yet, cannot save best checkpoint.")
            return
            
        current_val_dice = self.val_dice_scores[-1]
        if current_val_dice > self.best_dice_score + 1e-6: 
            checkpoint_path = f'{save_dir}/best_checkpoint_epoch_{self.epoch}.pth'
            optimizer_state = self.optimizer.state_dict() if self.optimizer else None
            current_train_loss = self.train_losses[-1] if self.train_losses else float('inf')
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'loss': current_train_loss,
                'dice_score': current_val_dice,
                'model_path': checkpoint_path
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved new best checkpoint to {checkpoint_path} with validation dice score: {current_val_dice:.4f}")

    def update_best_dice_score(self, save_dir):
        if not self.val_dice_scores:
            return
            
        current_val_dice = self.val_dice_scores[-1]
        if current_val_dice > self.best_dice_score + 1e-6:
            self.best_dice_score = current_val_dice
            self.best_dice_score_epoch = self.epoch
            self.best_dice_score_model_path = f'{save_dir}/best_checkpoint_epoch_{self.epoch}.pth' 

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
        
    def diameter_loss(self, outputs, labels):
        # This needs significant update if used with 1-channel logit output
        self.logger.warning("Diameter loss calculation is likely incorrect for 1-channel logit output.")
        return 0.0 
