import argparse
import logging
import torch
import sys
from pathlib import Path
import yaml

from richard.src.models.VNet2D import VNet2D # Adjust import if model is elsewhere

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_model_parameters(checkpoint_path: str):
    """Loads a model based on config and prints its named parameters."""
    try:
        # Load the main configuration file using the utility function
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        cfg = checkpoint['config']
        print(f"Checkpoint: {cfg}")

        model_cfg = cfg.get("model", {})
        if not model_cfg:
            logging.error("Model configuration section ('model') missing in config.")
            return

        # --- Instantiate Model (similar to train/finetune scripts) ---
        model_params = model_cfg.copy() # Work with a copy
        model_name = model_params.pop("name", 'VNet2D')

        if not model_name:
             logging.error("Model 'name' missing in model configuration.")
             return

        model = None
        if model_name == "VNet2D":
            try:
                # Ensure necessary parameters are present (example: in_channels, num_classes)
                # You might need to adjust this based on VNet2D's __init__ requirements
                if 'in_channels' not in model_params or 'out_channels' not in model_params:
                     logging.warning("VNet2D might require 'in_channels' and 'out_channels'. Using defaults 1, 1.")
                     model_params.setdefault('in_channels', 1)
                     model_params.setdefault('out_channels', 1)

                model = VNet2D(**model_params)
                logging.info(f"Successfully instantiated model: {model_name}")
            except Exception as e:
                logging.error(f"Error initializing {model_name} model: {e}", exc_info=True)
                return
        else:
             logging.error(f"Unsupported model name ('{model_name}') in config.")
             return

        # --- Print Named Parameters ---
        print(f"\n--- Named Parameters for {model_name} ---")
        count = 0
        for name, param in model.named_parameters():
            print(f"{name:<60} Requires Grad: {param.requires_grad:<6} Shape: {param.shape}")
            count += 1
        print(f"--- Total parameters: {count} ---")

        # --- Optional: Print Named Modules (useful for prefix freezing) ---
        print(f"\n--- Named Modules for {model_name} ---")
        module_count = 0
        for name, module in model.named_modules():
            # Skip the top-level module itself if it has a blank name
            if name:
                print(f"- {name}")
                module_count += 1
        print(f"--- Total modules: {module_count} ---")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List parameters and modules of a model defined in a config file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint file (e.g., model.pth) which defines the model.")

    args = parser.parse_args()
    list_model_parameters(args.checkpoint) 