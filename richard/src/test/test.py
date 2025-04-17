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
from datetime import datetime
from pathlib import Path
from typing import Optional

# Custom imports
from richard.src.data.dataset import AllDataset
from richard.src.utils.utils import dice_score, set_seed, setup_logging, load_config_decorator, set_visible_devices
from richard.src.utils.loss import CombinedLoss, SUPPORTED_LOSSES
from richard.src.models.VNet2D import VNet2D
from richard.src.test.tester import Tester

def select_checkpoint_interactive(checkpoint_dir_str: str) -> Optional[str]:
    """Lists .pth files in a directory and prompts the user for selection."""
    checkpoint_dir = Path(checkpoint_dir_str)
    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None

    try:
        # Sort by modification time, newest first
        checkpoints = sorted([f for f in checkpoint_dir.glob("*.pth") if f.is_file()], 
                             key=os.path.getmtime, reverse=True)
        checkpoint_names = [f.name for f in checkpoints]
    except Exception as e:
        print(f"Error listing checkpoints in {checkpoint_dir}: {e}")
        return None

    if not checkpoint_names:
        print(f"No checkpoint (.pth) files found in {checkpoint_dir}.")
        return None

    print("\nAvailable checkpoints (newest first):")
    for i, ckpt_name in enumerate(checkpoint_names):
        print(f"  [{i+1}] {ckpt_name}")

    while True:
        try:
            selection = input(f"Select checkpoint number (1-{len(checkpoint_names)}): ")
            index = int(selection) - 1
            if 0 <= index < len(checkpoint_names):
                selected_ckpt = checkpoint_names[index]
                print(f"Selected: {selected_ckpt}")
                return selected_ckpt
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(checkpoint_names)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None

@load_config_decorator(config_arg_name="config")
def test(args: argparse.Namespace, cfg: dict, checkpoint_basename: str):
    # Set device
    set_visible_devices(args.cuda_visible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Test function received config keys: {list(cfg.keys())}")

    # --- Create Unique Output Directory based on Checkpoint Name --- 
    test_cfg = cfg.get('testing', {}) # Get testing config section
    results_base_dir = Path(test_cfg.get('results_dir', 'richard/test_results'))
    output_dir = results_base_dir / checkpoint_basename # Use checkpoint name for subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Test results will be saved in: {output_dir}")

    # --- Dataset and DataLoader ---
    dataset_cfg = cfg.get("dataset", {})
    dataloader_cfg = cfg.get("dataloader", {})

    if not dataset_cfg or not dataset_cfg.get("test"):
        logging.error("Test dataset configuration missing or empty. Exiting.")
        return
    if not dataloader_cfg or not dataloader_cfg.get("test"):
        logging.error("Test dataloader configuration missing or empty. Exiting.")
        return

    try:
        test_dataset = AllDataset(**dataset_cfg.get("test", {}))
        logging.info(f"Loaded test dataset with {len(test_dataset)} samples.")
        test_loader = DataLoader(test_dataset, **dataloader_cfg.get("test", {}))
        logging.info("Created Test DataLoader.")
    except Exception as e:
        logging.error(f"Error creating Test Dataset or DataLoader: {e}", exc_info=True)
        return
    
    # --- Model ---
    model_params = cfg.get("model", {}).copy()
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

    # --- Load Model Checkpoint --- 
    # Construct path from config dir and command-line name
    checkpoint_dir = test_cfg.get("model_checkpoint_dir", None)
    if not checkpoint_dir:
        logging.error("'model_checkpoint_dir' missing in testing configuration.")
        return
        
    model_checkpoint_path = Path(checkpoint_dir) / args.ckpt_name
    
    if not model_checkpoint_path.exists():
        logging.error(f"Constructed model checkpoint path does not exist: {model_checkpoint_path}")
        return
        
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model state_dict from checkpoint: {model_checkpoint_path}")
            if 'epoch' in checkpoint:
                 logging.info(f"Checkpoint trained for {checkpoint['epoch']} epochs.")
        else:
             model.load_state_dict(checkpoint)
             logging.warning(f"Loaded state_dict directly from checkpoint file (expected 'model_state_dict' key): {model_checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading model checkpoint {model_checkpoint_path}: {e}", exc_info=True)
        return

    # --- Criterion Setup if Provided ---
    criterion_config_dict = cfg.get("criterion")
    combined_criterion = None
    if criterion_config_dict:
        try:
            combined_criterion = CombinedLoss(loss_config=criterion_config_dict, device=device)
            logging.info("Initialized CombinedLoss criterion for test loss calculation.")
        except Exception as e:
            logging.error(f"Error initializing CombinedLoss for testing: {e}", exc_info=True)
            combined_criterion = None
    else:
        logging.warning("Criterion configuration missing. Test loss will not be calculated.")
    
    # --- Tester Setup ---
    thresh_min = test_cfg.get('thresh_min', 0.0)
    thresh_max = test_cfg.get('thresh_max', 1.0)
    thresh_step = test_cfg.get('thresh_step', 0.05)
    num_visuals = test_cfg.get('num_visual_samples', 10)
    overlay_opacity = test_cfg.get('overlay_opacity', 0.4)
    
    tester = Tester(
        model=model,
        test_loader=test_loader,
        criterion=combined_criterion,
        device=device,
        output_dir=output_dir,
        thresh_min=thresh_min,
        thresh_max=thresh_max,
        thresh_step=thresh_step,
        num_visual_samples=num_visuals,
        overlay_opacity=overlay_opacity
    )
    
    # --- Run Evaluation and Visualization ---
    try:
        best_dice, best_threshold, _ = tester.evaluate()
        tester.generate_visualizations(best_threshold=best_threshold)
        logging.info("Testing process finished.")
    except Exception as e:
         logging.error(f"Error during tester execution: {e}", exc_info=True)

if __name__ == "__main__":
    # Set seed
    set_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Test a segmentation model.")
    parser.add_argument("--config", type=str, default="richard/config/test.yaml",
                        help="Path to the main testing configuration file.")
    parser.add_argument("--ckpt_name", type=str, required=False, default=None,
                        help="Filename of the checkpoint (.pth file) to test. If omitted, interactive selection is triggered.")
    parser.add_argument("--cuda_visible", type=str, required=False,
                        help="Comma-separated list of GPU IDs to use. (e.g. '0,1')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode logging.")
    args = parser.parse_args()
    
    # --- Determine Checkpoint Name (Interactive if needed) --- 
    if not args.ckpt_name:
        # Define the directory to search for checkpoints
        # This could also be read from the config *if* absolutely necessary,
        # but hardcoding or using a default is simpler as requested.
        checkpoint_dir = "/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/checkpoints"
        print(f"--ckpt_name not provided. Searching in: {checkpoint_dir}")
        selected_ckpt = select_checkpoint_interactive(checkpoint_dir)
        if selected_ckpt is None:
             print("No checkpoint selected. Exiting.")
             exit(0)
        args.ckpt_name = selected_ckpt # Update args with the selected name
    
    # Use checkpoint name (without extension) for log/output folder naming
    checkpoint_basename = Path(args.ckpt_name).stem 
    
    # --- Setup Logging --- 
    # Get log directory from config (will be loaded by decorator later)
    # For now, use a default or assume it exists.
    # A more robust way might involve a quick peek into the config file,
    # but let's stick to the default for simplicity.
    log_dir = "richard/logs" 
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(console_level=log_level, file_level=logging.DEBUG, 
                  log_dir=log_dir, 
                  start_time=checkpoint_basename, # Use basename for log file name part
                  test=True)
                  
    logging.info(f"Attempting to test checkpoint: {args.ckpt_name}")
    logging.info(f"Using config: {args.config}")

    # --- Run Test --- 
    # Config is loaded by the decorator on the 'test' function
    # Pass args (contains ckpt_name) and the derived basename
    try:
        test(args, checkpoint_basename=checkpoint_basename) 
    except Exception as e:
        logging.error(f"Unhandled exception during test execution: {e}", exc_info=True)
