import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Optional
import logging
import os

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: data.DataLoader, 
                 val_loader: data.DataLoader, 
                 test_loader: Optional[data.DataLoader]=None, 
                 optimizer: Optional[optim.Optimizer]=None, 
                 criterion: Optional[nn.Module]=None, 
                 device: Optional[torch.device]=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logging.getLogger(__name__)

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
    
    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.update_best_dice_score()
            self.save_checkpoint()
            self._log_epoch_metrics()
        # Save the final checkpoint regardless of performance
        final_checkpoint_path = f'Richard/checkpoints/final_checkpoint.pth'
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses[-1],
            'dice_score': self.val_dice_scores[-1],
            'model_path': final_checkpoint_path
        }, final_checkpoint_path)
        self.logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    def train_one_epoch(self):
        self.model.train()

        # Reset metrics
        running_loss = 0.0
        correct = 0
        total = 0
        dice_score = 0.0
        diameter_loss = 0.0

        # Train loop
        for i, (images, labels) in enumerate(self.train_loader, 0):
            # Move to device
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images) # type: torch.Tensor
            loss = self.criterion(outputs, labels) # type: torch.Tensor

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update running metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0) 
            correct += predicted.eq(labels).sum().item()
            dice_score += self.dice_score(outputs, labels)
            diameter_loss += self.diameter_loss(outputs, labels)

        # Update metrics
        self.train_losses.append(running_loss / (i + 1))
        self.train_dice_scores.append(dice_score / (i + 1))
        self.train_diameter_losses.append(diameter_loss / (i + 1))
    
    def validate(self):
        self.model.eval()

        # Reset metrics
        running_loss = 0.0
        correct = 0
        total = 0
        dice_score = 0.0
        diameter_loss = 0.0

        # Validate loop
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader, 0):
                # Move to device
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images) # type: torch.Tensor
                loss = self.criterion(outputs, labels) # type: torch.Tensor

                # Update running metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                dice_score += self.dice_score(outputs, labels)
                diameter_loss += self.diameter_loss(outputs, labels)

        # Update metrics
        self.val_losses.append(running_loss / (i + 1))
        self.val_dice_scores.append(dice_score / (i + 1))
        self.val_diameter_losses.append(diameter_loss / (i + 1))
    
    def test(self):
        self.model.eval()

        # Reset metrics
        running_loss = 0.0
        correct = 0
        total = 0
        dice_score = 0.0
        diameter_loss = 0.0

        # Test loop
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader, 0):
                # Move to device
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images) # type: torch.Tensor
                loss = self.criterion(outputs, labels) # type: torch.Tensor

                # Update running metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                dice_score += self.dice_score(outputs, labels)
                diameter_loss += self.diameter_loss(outputs, labels)

        # Update metrics
        self.test_losses.append(running_loss / (i + 1))
        self.test_dice_scores.append(dice_score / (i + 1))
        self.test_diameter_losses.append(diameter_loss / (i + 1))

        # Log test results
        self.logger.info(f"--- Testing Complete ---")
        self.logger.info(f"Test Loss: {self.test_losses[-1]:.4f}")
        self.logger.info(f"Test Dice Score: {self.test_dice_scores[-1]:.4f}")
        self.logger.info(f"Test Diameter Loss: {self.test_diameter_losses[-1]:.4f}")

    def save_checkpoint(self):
        # Only save checkpoint if it's better than previous best
        if self.val_dice_scores[-1] > self.best_dice_score:
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.train_losses[-1],
                'dice_score': self.val_dice_scores[-1],
                'model_path': f'Richard/checkpoints/best_checkpoint_epoch_{self.epoch}.pth'
            }
            checkpoint_path = f'Richard/checkpoints/best_checkpoint_epoch_{self.epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved new best checkpoint to {checkpoint_path} with validation dice score: {self.val_dice_scores[-1]:.4f}")

    def update_best_dice_score(self):
        if self.val_dice_scores[-1] > self.best_dice_score:
            self.best_dice_score = self.val_dice_scores[-1]
            self.best_dice_score_epoch = self.epoch
            self.best_dice_score_model_path = f'Richard/checkpoints/best_checkpoint_epoch_{self.epoch}.pth'

    def _log_epoch_metrics(self):
        """Logs metrics for the current epoch using the configured logger."""
        self.logger.info(f"--- Epoch {self.epoch} Summary ---")
        # Check if training metrics exist before logging
        if self.train_losses:
            self.logger.info(f"Train Loss: {self.train_losses[-1]:.4f}")
        if self.train_dice_scores:
            self.logger.info(f"Train Dice Score: {self.train_dice_scores[-1]:.4f}")
        if self.train_diameter_losses:
            self.logger.info(f"Train Diameter Loss: {self.train_diameter_losses[-1]:.4f}")

        # Check if validation metrics exist before logging
        if self.val_losses:
            self.logger.info(f"Val Loss: {self.val_losses[-1]:.4f}")
        if self.val_dice_scores:
            self.logger.info(f"Val Dice Score: {self.val_dice_scores[-1]:.4f}")
        if self.val_diameter_losses:
            self.logger.info(f"Val Diameter Loss: {self.val_diameter_losses[-1]:.4f}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def dice_score(self, outputs, labels):
        # Ensure outputs and labels are on the same device
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)
        
        # Flatten the outputs and labels
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        
        # Calculate Dice score
        intersection = (outputs * labels).sum()
        dice_score = (2 * intersection + 1e-6) / (outputs.sum() + labels.sum() + 1e-6)
        return dice_score
    
    def diameter_loss(self, outputs, labels):
        # Ensure outputs and labels are on the same device
        outputs = outputs.to(self.device)
        labels = labels.to(self.device)

        # Calculate diameter loss
        # This is a placeholder for the actual diameter loss calculation
        diameter_loss = 0.0
        return diameter_loss
