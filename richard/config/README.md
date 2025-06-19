# Configuration Guide

This document explains the configuration system used for training, fine-tuning, and testing models. The system uses YAML files for easy management of hyperparameters and settings.

## Core Concept

The configuration is modular. There are three main entry-point configuration files:
- `train.yaml`: For training a model from scratch.
- `finetune.yaml`: For fine-tuning a pre-trained model.
- `test.yaml`: For evaluating a trained model.

These main files reference smaller, component-specific configuration files located in the subdirectories (`model/`, `dataset/`, etc.). The scripts (`train.py`, `finetune.py`, `test.py`) load the main config file, which then directs the loader to pull in the specified component configs and merge them into a single configuration dictionary.

For example, in `train.yaml`, the line `model: vnet` tells the system to load the configuration from `config/model/vnet.yaml` and place its contents under the `model` key in the final configuration object.

## Main Configuration Files

### `train.yaml`

Used by `richard/src/train/train.py` to train a new model.

- **`model`**: Specifies the model architecture. E.g., `vnet` maps to `model/vnet.yaml`.
- **`dataset`**: Specifies the dataset for training and validation. E.g., `splits_2d` maps to `dataset/splits_2d.yaml`.
- **`dataloader`**: Specifies data loading parameters. E.g., `dataloader` maps to `dataloader/dataloader.yaml`.
- **`optimizer`**: Specifies the optimizer. E.g., `AdamW` maps to `optimizer/AdamW.yaml`.
- **`scheduler`**: Specifies the learning rate scheduler. E.g., `ReduceLROnPlateau` maps to `scheduler/ReduceLROnPlateau.yaml`.
- **`criterion`**: Defines the loss function(s) directly. This section is not a reference to another file but contains the loss configuration itself. You can combine multiple losses.
- **`training`**: Contains training-specific parameters:
    - `num_epochs`: Total number of epochs to train for.
    - `early_stopping_patience`: Number of epochs to wait for improvement in validation Dice score before stopping.
    - `early_stopping_min_delta`: The minimum change in the monitored quantity to qualify as an improvement.

### `finetune.yaml`

Used by `richard/src/train/finetune.py`. It inherits all the keys from `train.yaml` and adds a `checkpoint` section.

- **`checkpoint`**:
    - `path`: Absolute path to the pre-trained model checkpoint (`.pth` file).
    - `freeze_config`:
        - `freeze`: `true` to freeze layers, `false` to train all layers.
        - `config`: A dictionary where keys are the names of model layers/blocks (as defined in the `VNet2D` class) and values are `true` to freeze or `false` to train. The freezing logic supports prefix matching (e.g., `"down_blocks": true` will freeze all layers starting with that name).

> **Note**: To find the correct layer names for freezing, you should inspect the `named_parameters()` of the model in `richard/src/models/VNet2D.py`. The names in the default `finetune.yaml` (e.g., `downsample_block_1`) may not match the actual model structure, which can cause freezing to fail silently.

### `test.yaml`

Used by `richard/src/test/test.py` for evaluation.

- **`model`**, **`dataset`**, **`dataloader`**: Similar to `train.yaml`, specifies the components for testing.
- **`testing`**:
    - `model_checkpoint_dir`: Directory where trained model checkpoints are stored. The test script will look for the checkpoint specified by the `--ckpt_name` argument here.
    - `results_dir`: Base directory to save testing results (like masks, overlays, and logs).
    - `thresh_min`, `thresh_max`, `thresh_step`: Defines the range of probability thresholds to test for finding the best Dice score.
    - `num_visual_patients`: Number of patients for whom to save visualization overlays. Use `-1` to visualize all patients in the test set.
    - `overlay_opacity`: Opacity of the predicted mask when overlaid on the original image.

## Component Configuration Files

These files are located in subdirectories and define the parameters for specific components of the pipeline.

### `criterion/`

Defines loss functions. The main config's `criterion` section is where you combine them.

**Example of a combined loss in `train.yaml`:**
```yaml
criterion:
    Dice:
        name: DiceLoss
        weight: 1.0
```
- **`Dice`**: A custom key for this loss component. You can name it anything.
- **`name`**: The class name of the loss function. Must be one of the supported losses in `richard/src/utils/loss.py` (e.g., `BCEWithLogitsLoss`, `DiceLoss`, `FocalLoss`).
- **`weight`**: The weight of this loss in the final combined loss. If weights for all components are specified and don't sum to 1, they will be normalized.
- **`params`**: An optional dictionary for loss-specific parameters. For example, `FocalLoss` takes `alpha` and `gamma`, and `BCEWithLogitsLoss` can take `pos_weight`.

### `dataloader/dataloader.yaml`

Specifies settings for `torch.utils.data.DataLoader`.
- **`train`**, **`val`**, **`test`**: Sections for different data splits.
    - `batch_size`: Number of samples per batch.
    - `shuffle`: `true` to randomly shuffle data every epoch. Should be `true` for training.
    - `num_workers`: Number of subprocesses to use for data loading.

### `dataset/`

Defines the data sources.
- **`train`**, **`val`**, **`test`**: Sections for different data splits.
    - `image_dir`: Path to the directory with image files (`.npy`).
    - `mask_dir`: Path to the directory with corresponding mask files (`.npy`).
    - `augment`: `true` to apply data augmentation during training.
    - `normalize`: `true` to normalize the images.
    - `mean`, `std`: A list containing the mean and standard deviation values for normalization.

### `model/vnet.yaml`

Defines the hyperparameters for the `VNet2D` model.
- `name`: Should be `VNet2D` to match the model class used in the scripts.
- `in_channels`: Number of input channels (e.g., 1 for grayscale).
- `out_channels`: Number of output channels (e.g., 1 for binary segmentation).
- `dropout_rate`: Dropout probability.
- ... and other model-specific parameters. Refer to `richard/src/models/VNet2D.py` for a full list of constructor arguments.

### `optimizer/`

Defines the optimizer and its parameters.
- `name`: The optimizer to use (e.g., `AdamW` or `Adam`).
- `lr`: Learning rate.
- `weight_decay`: Weight decay (L2 penalty).

### `scheduler/`

Defines the learning rate scheduler.
- `name`: The scheduler to use (e.g., `ReduceLROnPlateau`, `CosineAnnealingLR`, or `StepLR`).
- Contains scheduler-specific parameters, such as `factor`, `patience`, and `min_lr` for `ReduceLROnPlateau`.

## How to Customize

### Example 1: Add a BCE Loss to the training

To train with both Dice and BCE loss, modify the `criterion` section in `train.yaml` or `finetune.yaml`:
```yaml
criterion:
    Dice:
        name: DiceLoss
        weight: 0.5 # Split the weight
    BCE:
        name: BCEWithLogitsLoss
        weight: 0.5 # Split the weight
        # Optional: give more weight to the positive class for imbalanced datasets
        params:
            pos_weight: 102.0
```

### Example 2: Change the model's dropout rate

1.  Open `richard/config/model/vnet.yaml`.
2.  Modify the `dropout_rate` value:
    ```yaml
    dropout_rate: 0.5 # Changed from 0.3
    ```

### Example 3: Freeze the encoder part of the VNet for fine-tuning

1.  Open `richard/config/finetune.yaml`.
2.  Modify the `checkpoint.freeze_config` section. You need to know the layer names from your model definition.
    ```yaml
    freeze_config:
        freeze: true
        config:
            # These names must match the prefixes of parameter names
            # from model.named_parameters() in PyTorch.
            "init_conv": true
            "down_blocks": true # Freezes all layers starting with "down_blocks"
            "final_conv": false # Ensure the final layer is trainable
    ```
The freeze logic uses `startswith`, so providing `"down_blocks": true` will freeze `down_blocks.0`, `down_blocks.1`, etc., which is a convenient way to freeze entire sections of the network. 