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
from typing import Union, Optional
from datetime import datetime
import json
import copy # Import the copy module

# Assuming these modules exist based on previous context
from richard.src.data.dataset import AllDataset
from richard.src.models.VNet2D import VNet2D
from richard.src.train.trainer import Trainer
from richard.src.utils.utils import setup_logging, set_visible_devices, set_seed, load_config_decorator
from richard.src.utils.loss import CombinedLoss

# Apply the decorator
@load_config_decorator(config_arg_name="config")
def train(args: argparse.Namespace, cfg: dict, start_time: Optional[str]=None): # Add cfg parameter
    # Set device
    set_visible_devices(args.cuda_visible) # Note: This might interact oddly with the decorator if it needs args.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # The 'cfg' dictionary is directly available.
    logging.info(f"Train function received config keys: {list(cfg.keys())}")

    # Create a deep copy of the config for modification during instantiation
    cfg_for_instantiation = copy.deepcopy(cfg)
    logging.debug("Created deep copy of config for instantiation.")

    # --- Dataset and DataLoader ---
    # Use the injected cfg dictionary
    dataset_cfg = cfg_for_instantiation.get("dataset", {})
    dataloader_cfg = cfg_for_instantiation.get("dataloader", {})

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
    model_params = cfg_for_instantiation.get("model", {})
    if not model_params:
        logging.error("Model configuration missing. Exiting.")
        return
    model_name = model_params.pop("name", None)

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

    # --- Load Checkpoint ---
    checkpoint_params = cfg_for_instantiation.get("checkpoint", {}) #type: dict
    if checkpoint_params:
        checkpoint_path = checkpoint_params.get("path", None)
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                # Store the original training config from the checkpoint if needed
                checkpoint_config = checkpoint.get("config") # Optional: get original config
                logging.info(f"Loaded model state_dict from checkpoint: {checkpoint_path}")
                if checkpoint_config:
                     logging.debug(f"Original Model Training Config: {json.dumps(checkpoint_config, indent=2, sort_keys=False)}")
                else:
                     logging.warning("Checkpoint did not contain a 'config' key.")

            except FileNotFoundError:
                 logging.error(f"Checkpoint file not found: {checkpoint_path}")
                 return
            except Exception as e:
                 logging.error(f"Error loading checkpoint from {checkpoint_path}: {e}", exc_info=True)
                 return
        else:
            logging.error("No checkpoint path provided in config. Exiting.")
            return
        
        # --- Freeze Layers ---
        freeze_config = checkpoint_params.get("freeze_config", {})
        freeze = freeze_config.get("freeze", False)
        if freeze:
            freeze_config_dict = freeze_config.get("config", {})
            if freeze_config_dict:
                frozen_params = []
                for name, param in model.named_parameters():
                    # Check if the parameter name (or its prefix for layers) is marked for freezing
                    if name in freeze_config_dict and freeze_config_dict[name]:
                        param.requires_grad = False
                        frozen_params.append(name)
                    # Allow freezing entire layers/blocks by prefix
                    elif any(name.startswith(prefix) for prefix in freeze_config_dict if freeze_config_dict[prefix]):
                        param.requires_grad = False
                        frozen_params.append(name)

                if frozen_params:
                    logging.info(f"Frozen parameters based on freeze_config: {frozen_params}")
                else:
                    logging.warning(f"Freeze was enabled, but no parameters matched the names in freeze_config: {list(freeze_config_dict.keys())}")
            else:
                logging.error("Freeze config 'config' dictionary not provided or empty when freeze is True. Exiting.")
                return
        else:
            logging.info("Unfrozen all layers.")
    else:
        logging.error("No checkpoint provided. Exiting.")
        return

    # --- Optimizer ---
    optimizer_defaults = {"name": "Adam", "lr": 1e-4}
    optimizer_cfg = cfg_for_instantiation.get("optimizer", optimizer_defaults)
    optimizer_name = optimizer_cfg.pop("name")

    optimizer = None
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_cfg)
        logging.info(f"Initialized Adam optimizer with params: {optimizer_cfg}")
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), **optimizer_cfg)
        logging.info(f"Initialized AdamW optimizer with params: {optimizer_cfg}")
    else:
        logging.error(f"Unsupported optimizer: {optimizer_name}")
        return
    
    # --- Scheduler ---
    scheduler_cfg = cfg_for_instantiation.get("scheduler", None)
    scheduler = None # Initialize scheduler to None
    if scheduler_cfg:
        scheduler_name = scheduler_cfg.pop("name")
        
        if scheduler_name.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg)
            logging.info(f"Initialized StepLR scheduler with params: {scheduler_cfg}")

        elif scheduler_name.lower() == "cosineannealinglr":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg)
            logging.info(f"Initialized CosineAnnealingLR scheduler with params: {scheduler_cfg}")

        elif scheduler_name.lower() == "reducelronplateau":
            # Ensure mode is set correctly if using ReduceLROnPlateau with Dice
            scheduler_cfg.setdefault('mode', 'max') # Default to 'max' for Dice score
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
            logging.info(f"Initialized ReduceLROnPlateau scheduler with params: {scheduler_cfg}")

        else:
            logging.error(f"Unsupported scheduler: {scheduler_name}")
            return
    else:
        logging.info("No scheduler configuration provided.")
        
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
        scheduler=scheduler,
        criterion=combined_criterion,
        device=device,
        config=cfg,
        start_time=start_time
    )

    logging.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs, save_dir=args.save_dir)
    logging.info("Training finished.")

if __name__ == "__main__":
    # Set seed
    set_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", type=str, default="richard/config/finetune.yaml",
                        help="Path to the main finetune configuration file.")
    parser.add_argument("--cuda_visible", type=str, required=False,
                        help="Comma-separated list of GPU IDs to use. (e.g. '0,1')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (less verbose logging).")
    parser.add_argument("--save_dir", type=str, default="richard/checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--log_dir", type=str, default="richard/logs",
                        help="Directory to save logs.")
    args = parser.parse_args()

    start_time = datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    # Set logging
    if args.debug:
        setup_logging(console_level=logging.DEBUG, file_level=logging.DEBUG, log_dir=args.log_dir, start_time=start_time)
    else:
        setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_dir=args.log_dir, start_time=start_time)

    logging.info(f"Start time: {start_time}")
    logging.info(f"This is a finetune script.")

    # Train
    # Call train as before; decorator handles config loading and injection
    try:
        train(args, start_time=start_time)
    except Exception as e:
        logging.error(f"Unhandled exception during train execution: {e}", exc_info=True)