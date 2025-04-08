import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import yaml

from Richard.dataset import AllDataset
from Richard.utils import load_model, dice_score, set_seed

BASE_DIR = "/radraid2/dongwoolee/VNet_LungNoduleSeg/Richard"

print(os.getcwd())
def main(config):
    # Extract parameters from config
    eval_config = config['evaluation']
    image_dir = eval_config['image_dir']
    mask_dir = eval_config['mask_dir']
    model_path = eval_config['model_path']
    cuda_devices = str(eval_config['cuda_devices'])
    batch_size = eval_config['batch_size']
    seed = eval_config['seed']
    
    # Set seed
    set_seed(seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if torch.cuda.is_available() and cuda_devices != "-1":
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {cuda_devices}")
    else:
        device = torch.device("cpu")
        print("CUDA not available or disabled, using CPU.")

    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device, eval=True)
    print(f"Model loaded: {model.__class__.__name__}")

    # Create dataset and dataloader
    print(f"Loading dataset from Image Dir: {image_dir}, Mask Dir: {mask_dir}")
    dataset = AllDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize list to store Dice scores for each batch
    all_dice_scores = []

    # Inference loop
    model.eval()
    with torch.no_grad():
        # Wrap dataloader with tqdm for progress bar
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for i, (images, masks) in enumerate(progress_bar):
            # Move data to the specified device
            images = images.to(device) # type: torch.Tensor
            masks = masks.to(device) # type: torch.Tensor

            # Make predictions (outputs have shape [N, 2, D, H, W])
            outputs = model(images)
            
            # Get predicted class indices by taking argmax along the channel dimension
            # Shape becomes [N, D, H, W] with values 0 or 1
            preds = torch.argmax(outputs, dim=1)

            # Ensure preds and masks are the same data type for dice calculation
            # (Dice score function expects float)
            preds = preds.float()
            masks = masks.float()

            # Calculate Dice score for the batch
            batch_dice = dice_score(preds, masks)
            all_dice_scores.append(batch_dice.item())

            # Update tqdm progress bar description with current batch Dice score (optional)
            progress_bar.set_postfix({"Batch Dice": f"{batch_dice.item():.4f}"})

    # Calculate and print the average Dice score
    average_dice = np.mean(all_dice_scores)
    print(f"\nAverage Dice Score over the dataset: {average_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VNet model using a configuration file.")
    
    parser.add_argument('--config', type=str, default="config_LUNA16_val.yaml", 
                        help='Path to the YAML configuration file.')

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(os.path.join(BASE_DIR, args.config), 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {args.config}: {e}")
        exit(1)

    # Run the main evaluation function with the loaded config
    main(config)