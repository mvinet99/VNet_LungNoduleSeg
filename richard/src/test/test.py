import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import logging

# Custom imports
from richard.src.data.dataset import AllDataset
from richard.src.utils.utils import load_model, dice_score, set_seed, setup_logging, load_config_decorator, set_visible_devices
from richard.src.train.trainer import Trainer
from richard.src.utils.loss import CombinedLoss
from richard.src.models.VNet2D import VNet2D

@load_config_decorator(config_arg_name="config")
def test(args: argparse.Namespace, cfg: dict):
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
        test_dataset = AllDataset(**dataset_cfg.get("test", {}))
        logging.info(f"Loaded test dataset with {len(test_dataset)} samples.")

        test_loader = DataLoader(test_dataset, **dataloader_cfg.get("test", {}))
        logging.info("Created DataLoader.")
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

    # --- Load Model Checkpoint ---
    model_checkpoint = cfg.get("model_checkpoint", None)
    if not model_checkpoint:
        logging.error("Model checkpoint missing. Exiting.")
        return
    # Load the full checkpoint dictionary first
    checkpoint = torch.load(model_checkpoint[0], weights_only=False) 
    # Load the model state dictionary from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model state_dict from checkpoint: {model_checkpoint[0]}")

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
    
    # --- Test ---
    trainer = Trainer(model=model, test_loader=test_loader, device=device, criterion=combined_criterion)
    trainer.test()

if __name__ == "__main__":
    # Set seed
    set_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Test a segmentation model.")
    parser.add_argument("--config", type=str, default="richard/config/test.yaml",
                        help="Path to the main testing configuration file.")
    parser.add_argument("--cuda_visible", type=str, required=False,
                        help="Comma-separated list of GPU IDs to use. (e.g. '0,1')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (less verbose logging).")
    args = parser.parse_args()

    # Set logging
    if args.debug:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)

    # Test
    try:
        test(args)
    except Exception as e:
        logging.error(f"Unhandled exception during test execution: {e}", exc_info=True)
        raise e
