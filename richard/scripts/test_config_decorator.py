import os
import sys
import argparse
import logging
# Removed shutil as it's not needed now
from pathlib import Path


from richard.src.utils.utils import load_config_decorator, setup_logging

# --- Setup Logging ---
setup_logging(level=logging.INFO) 
logger = logging.getLogger(__name__)

# --- Test Function to Receive Config ---
# Apply the decorator, using the standard 'config' argument name
@load_config_decorator(config_arg_name="config") 
def check_loaded_config(args: argparse.Namespace, cfg: dict):
    """This function only receives args and the config loaded by the decorator."""
    logger.info("Inside check_loaded_config!")
    logger.info(f"Received args: {args}")
    logger.info("Received merged cfg dictionary from decorator:")
    import json # For pretty printing the dict
    # Use ensure_ascii=False if you have non-ASCII characters
    logger.info(json.dumps(cfg, indent=2, ensure_ascii=False))
    logger.info("Config check finished.")

# --- Main Test Logic --- 
if __name__ == "__main__":
    
    # --- Use Argparse like in train.py --- 
    parser = argparse.ArgumentParser(
        description="Test script for the configuration loading decorator. "
                    "Loads the specified config and prints the merged result."
    )
    # Argument name '--config' corresponds to args.config used by decorator
    parser.add_argument("--config", type=str, default="richard/config/train.yaml",
                        help="Path to the main configuration file to test loading.")
    # Add other arguments if the decorator or test function needs them, 
    # but keep it simple for just testing config loading.
    # parser.add_argument("--some_other_arg", default="value") 
    
    args = parser.parse_args()
    
    logger.info("-"*20 + f" Testing config loading for: {args.config} " + "-"*20)
    try:
        # Call the decorated function with the parsed arguments.
        # The decorator will handle loading based on args.config and inject 'cfg'.
        check_loaded_config(args)
        logger.info("Decorator test call completed successfully.")
    except FileNotFoundError as e:
        # Specifically catch errors related to missing config files
        logger.error(f"Config loading failed: {e}", exc_info=False) # No need for full traceback here
    except Exception as e:
        # Catch other potential errors during decorator execution or function call
        logger.error(f"An error occurred during the test call: {e}", exc_info=True)
    logger.info("-"*20 + " Test call finished " + "-"*20)