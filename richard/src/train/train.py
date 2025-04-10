import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
from typing import Union
# Assuming these modules exist based on previous context
from richard.src.data.dataset import AllDataset
from richard.src.models.VNet2D import VNet2D
from richard.src.train.trainer import Trainer
from richard.src.utils.utils import setup_logging, set_visible_devices, set_seed, load_config_decorator
from richard.src.utils.loss import CombinedLoss

# Import DiceLoss if it's defined in utils
# Ensure DiceLoss is imported if you intend to use it
# from richard.src.utils.utils import DiceLoss 

# Apply the decorator
@load_config_decorator(config_arg_name="config")
def train(args: argparse.Namespace, cfg: dict): # Add cfg parameter
    # Set device
    set_visible_devices(args.cuda_visible) # Note: This might interact oddly with the decorator if it needs args.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # The 'cfg' dictionary is directly available.
    logging.info(f"Train function received config keys: {list(cfg.keys())}")

    # --- Dataset and DataLoader ---
    # Use the injected cfg dictionary
    dataset_cfg = cfg.get("dataset", {})
    dataloader_cfg = cfg.get("dataloader", {})

    if not dataset_cfg:
        logging.error("Dataset configuration missing or empty after loading. Exiting.")
        return
    if not dataloader_cfg:
        logging.error("Dataloader configuration missing or empty after loading. Exiting.")
        return

    try:
        train_dataset = AllDataset(**dataset_cfg.get("train", {}))
        val_dataset = AllDataset(**dataset_cfg.get("val", {}))
        logging.info(f"Loaded train dataset with {len(train_dataset)} samples.")
        logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")

        train_loader = DataLoader(train_dataset, **dataloader_cfg.get("train", {}))
        val_loader = DataLoader(val_dataset, **dataloader_cfg.get("val", {}))
        logging.info("Created DataLoaders.")
    except Exception as e:
        logging.error(f"Error creating Datasets or DataLoaders: {e}", exc_info=True)
        return

    # --- Model ---
    model_params = cfg.get("model", {})
    if not model_params:
        logging.error("Model configuration missing. Exiting.")
        return
    model_name = model_params.pop("name", None) # Get and remove name if exists

    if model_name == "VNet2D":
        try:
            model = VNet2D(**model_params).to(device)
            logging.info(f"Initialized model: VNet2D")
        except Exception as e:
            logging.error(f"Error initializing VNet2D model: {e}", exc_info=True)
            return
    else:
         logging.error(f"Unsupported model name ('{model_name}') or name missing in config.")
         return

    # --- Optimizer ---
    optimizer_cfg = cfg.get("optimizer", {"name": "Adam", "lr": 1e-4})
    optimizer_name = optimizer_cfg.pop("name")

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_cfg)
        logging.info(f"Initialized Adam optimizer with params: {optimizer_cfg}")
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), **optimizer_cfg)
        logging.info(f"Initialized AdamW optimizer with params: {optimizer_cfg}")
    else:
        logging.error(f"Unsupported optimizer: {optimizer_name}")
        return
        
    # --- Criterion Setup (Simplified) ---
    criterion_config_dict = cfg.get("criterion") # Get the dict (e.g., content of bce_dice.yaml)
    if not criterion_config_dict:
        logging.error("Criterion configuration section is missing or empty.")
        return
        
    try:
        # Instantiate the single CombinedLoss module
        combined_criterion = CombinedLoss(loss_config=criterion_config_dict, device=device)
        logging.info("Initialized CombinedLoss criterion.")
    except Exception as e:
        logging.error(f"Error initializing CombinedLoss: {e}", exc_info=True)
        return

    # --- Trainer Setup (Simplified) ---
    training_cfg = cfg.get("training", {"num_epochs": 30})
    num_epochs = training_cfg.get("num_epochs")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=combined_criterion,
        device=device
    )
    logging.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs, save_dir=args.save_dir)
    logging.info("Training finished.")

if __name__ == "__main__":
    # Set seed
    set_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", type=str, default="richard/config/train.yaml",
                        help="Path to the main training configuration file.")
    parser.add_argument("--cuda_visible", type=str, required=False,
                        help="Comma-separated list of GPU IDs to use. (e.g. '0,1')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (less verbose logging).")
    parser.add_argument("--save_dir", type=str, default="/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/checkpoints",
                        help="Directory to save checkpoints and logs.")
    args = parser.parse_args()

    # Set logging
    if args.debug:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)

    # Train
    # Call train as before; decorator handles config loading and injection
    try:
        train(args)
    except Exception as e:
        logging.error(f"Unhandled exception during train execution: {e}", exc_info=True)