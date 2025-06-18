import torch
import os
from pathlib import Path
import pprint

checkpoint_dir = Path("/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/checkpoints")
configs_found = {}

print(f"Checking directory: {checkpoint_dir}", flush=True)

if checkpoint_dir.is_dir():
    for filename in checkpoint_dir.glob("*.pth"):
        if filename.is_file():
            try:
                # Load the checkpoint. weights_only=False is needed to load potentially pickled config dicts
                checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
                # Check if it's a dictionary and contains the 'config' key
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    configs_found[str(filename)] = checkpoint['config']
                # else:
                #    print(f"'config' key missing in {filename}", flush=True) # Optional: uncomment to see missing files
            except Exception as e:
                print(f"Error processing {filename}: {e}", flush=True)

    if configs_found:
        print("\n--- Configurations found in checkpoints ---")
        for filename, config_val in configs_found.items():
            print(f"\nFile: {filename}")
            print("Config:")
            pprint.pprint(config_val, indent=4)
            print("-" * 20)
    else:
        print("\nNo .pth files found containing the 'config' key in the specified directory.")
else:
    print(f"Error: Directory not found: {checkpoint_dir}") 