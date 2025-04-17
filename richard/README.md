# Richard Module

This directory contains the primary code, configuration, and related files for the VNet Lung Nodule Segmentation project.

## Directory Structure

*   `config/`: Contains configuration files (YAML format) for different aspects of the project, such as model parameters, dataset paths, dataloader settings, training parameters, and loss functions.
    *   `train.yaml`: The main configuration file that references specific component configurations.
    *   Subdirectories (`model/`, `dataset/`, `dataloader/`, `criterion/`, `optimizer/`) hold the specific component YAML files.
    *   Subdirectory `scheduler/` will be added in the future version for training optimization
*   `checkpoints/`: Intended location for storing trained model weights. Stores both the model weights with best validation dice score and from final epoch.
Filename schema follows either `final_checkpoint_yy-mm-dd_hh:mm:ss.pth` or `best_checkpoint_yy-mm-dd_hh:mm:ss_epoch_n.pth`. The model weights (best and final) from the same script has the same timestamp (`yy-mm-dd_hh:mm:ss`).
*   `data/`: Intended location for storing datasets (e.g., LUNA16). Specific subdirectories might be used for raw, processed, train/validation splits, etc. (Note: Actual data might be large and stored elsewhere, with paths specified in config files).
*   `logs/`: Intended location for storing all training and testing logs. All saved logs have "debug" level logging. All logs filename shares the same timestamp as the model weights (e.g., `train_yy-mm-dd_hh:mm:ss.log`, `test_final_checkpoint_yy-mm-dd_hh:mm:ss.log`, `test_best_checkpoint_yy-mm-dd_hh:mm:ss_epoch_n.log`).
*   `scripts/`: Contains utility scripts, such as testing specific components (e.g., `test_config_decorator.py`) or data preprocessing scripts.
*   `src/`: Contains the core source code for the project.
    *   `data/`: Dataset loading and preprocessing logic (`dataset.py`).
    *   `models/`: Model architecture definitions (e.g., `VNet2D.py`).
    *   `train/`: Training loop and logic (`trainer.py`, `train.py`).
    *   `utils/`: Utility functions, helper classes (e.g., `loss.py`, `utils.py`), decorators, etc.
    *   `test/`: Testing lopp and logic (`tester.py`, `test.py`)
*   `checkpoints/`: Default directory for saving model checkpoints during and after training. (This directory is added to `.gitignore`).
*   `results/`: Intended directory for saving evaluation results, prediction outputs, etc. 
    *   `model_weight_filename/overlays/`: Stores all overlay of image, true mask and predicted mask from `test.py`.
    *   `model_weight_filename/predicted_masks/`: Stores all predicted mask for `test.py` with threshold maximizing test set dice score.

## Usage

The main entry point for training the model is typically:

```bash
# For Training - change the train.yaml accordingly

# Run from the project root (VNet_LungNoduleSeg)
python -m richard.src.train.train --config richard/config/train.yaml
```

```bash
# For Testing
# Run from the project root (VNET_LungNoduleSeg)
# Test scipts have interactive model weight choosing tool. Need model weights saved in the correct location to run the test script.
python -m richard.src.test.test --config richard/config/test.yaml
```

Refer to the specific configuration files in `config/` to adjust model hyperparameters, dataset paths, and training settings. 