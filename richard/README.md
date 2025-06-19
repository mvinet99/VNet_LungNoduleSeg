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
    *   `train/`: Training loop and logic (`trainer.py`, `train.py`, `finetune.py`).
    *   `utils/`: Utility functions, helper classes (e.g., `loss.py`, `utils.py`), decorators, etc.
    *   `test/`: Testing lopp and logic (`tester.py`, `test.py`)
*   `checkpoints/`: Default directory for saving model checkpoints during and after training. (This directory is added to `.gitignore`).
*   `results/`: Intended directory for saving evaluation results, prediction outputs, etc. 
    *   `model_weight_filename/overlays/`: Stores all overlay of image, true mask and predicted mask from `test.py`.
    *   `model_weight_filename/predicted_masks/`: Stores all predicted mask for `test.py` with threshold maximizing test set dice score.

## Usage

## Training Flow

To train a model, follow these steps:

1.  **Configure Your Training Run:**
    *   Navigate to the `richard/config/` directory.
    *   The main configuration file is `train.yaml`. This file specifies which component configurations to use for the run (e.g., model, dataset, optimizer, criterion).
    *   **Example `train.yaml` configuration:**
        *   **`model`, `dataset`, `dataloader`, `optimizer`, `scheduler`**: These keys link to the specific `.yaml` files in the corresponding subdirectories of `config/` that define the components for the run.
        *   **`criterion`**: Defines the loss function(s). You can combine multiple losses (e.g., `BCE` and `Dice`) and assign a `weight` to each. The trainer will use a `CombinedLoss` module.
        *   **`training`**: Contains hyperparameters for the training loop, such as `num_epochs`, `batch_size`, and parameters for early stopping.
    *   You can customize the training by editing the component-specific YAML files in the subdirectories (`model/`, `dataset/`, etc.) or by creating new ones and updating `train.yaml` to point to them. For example, to change the learning rate, you would edit the appropriate file in the `optimizer/` directory.

2.  **Execute the Training Script:**
    *   Run the following command from the root directory of the project (`VNet_LungNoduleSeg/`):

    ```bash
    python -m richard.src.train.train --config richard/config/train.yaml
    ```
    *   **Optional Arguments:**
        *   `--cuda_visible '0,1'`: Specify which GPU(s) to use.
        *   `--save_dir /path/to/checkpoints`: Override the default checkpoint directory.
        *   `--log_dir /path/to/logs`: Override the default log directory.
        *   `--debug`: Enable more verbose debug-level logging for both console and file output.

3.  **Monitor and Retrieve Results:**
    *   **Logs:** Live training progress and detailed logs are saved to the directory given as as argument for training script (`/path/to/logs`). Each training run generates a log file named `train_<timestamp>.log`.
    *   **Checkpoints:** The best-performing model checkpoint (based on validation Dice score) and the final model from the last epoch are saved in the `richard/checkpoints/` directory. The filenames include the corresponding timestamp (e.g., `best_checkpoint_<timestamp>_epoch_X.pth` and `final_checkpoint_<timestamp>.pth`), making it easy to link logs with models.

## Finetuning Flow

Finetuning allows you to take a pre-trained model and continue training it, which is ideal for adapting a model to a new dataset or for targeted training of specific layers.

1.  **Configure Your Finetuning Run:**
    *   Navigate to the `richard/config/` directory and open `finetune.yaml`.
    *   **Example `finetune.yaml` configuration:**
        The `finetune.yaml` file inherits most settings from `train.yaml` but adds a crucial `checkpoint` section:
        *   **`checkpoint.path`**: **This is the most important setting.** You must provide the full path to the pre-trained model checkpoint (`.pth` file) you want to finetune.
        *   **`checkpoint.freeze_config.freeze`**: A boolean (`true` or `false`) that acts as a master switch to enable or disable layer freezing.
        *   **`checkpoint.freeze_config.config`**: A dictionary where you list the names of model layers or blocks and set their value to `true` to freeze them. Any parameter whose name starts with one of these prefixes will not be updated during training.
    *   **Specify the Checkpoint:** In the `checkpoint` section, set the `path` to the pre-trained model checkpoint (`.pth` file) you want to load.
    *   **Configure Layer Freezing:** In the `freeze_config` section:
        *   Set `freeze: true` to free a certain layer in the VNet2D.

2.  **Execute the Finetuning Script:**
    *   Run the following command from the project's root directory:

    ```bash
    python -m richard.src.train.finetune --config richard/config/finetune.yaml
    ```
    *   The same optional arguments as the main training script (`--cuda_visible`, `--save_dir`, etc.) can be used.

3.  **Monitor and Retrieve Results:**
    *   The finetuning run will generate its own set of logs and checkpoints with a new timestamp, which will be saved in the same `logs` and `checkpoints` directories.

## Testing Flow

The test script evaluates a trained model on the test dataset. Its main purpose is to find the best possible Dice score by searching for the optimal probability threshold and to generate visual results for analysis.

1.  **Configure Your Test Run:**
    *   Navigate to `richard/config/` and open `test.yaml`.
    *   **Example `test.yaml` configuration:**
        *   The `model`, `dataset`, and `dataloader` keys should point to the configurations for your test run.
        *   The `testing` section contains parameters specific to the evaluation process:
            *   `model_checkpoint_dir`: Specifies the directory where the script will look for your saved model checkpoints (used for interactive selection).
            *   `results_dir`: Defines the base directory where all output folders (containing masks and overlays) will be created.
            *   `thresh_min`, `thresh_max`, `thresh_step`: Control the range and granularity of the search for the optimal Dice score threshold.
            *   `num_visual_patients`: The number of patients for which to save visual overlay images. Set to `-1` to visualize all patients in the test set.
    *   Ensure the `dataset` section points to your test data.
    *   Confirm that `model_checkpoint_dir` points to the directory where your trained models are saved (e.g., `richard/checkpoints`).
    *   Confirm that `results_dir` points to your desired directory to store your resulting masks and overaly images.

2.  **Execute the Test Script:**
    *   Run the script from the project's root directory. You can select a checkpoint in one of two ways:

    **A) Interactive Selection (Recommended):**
    Simply run the command without specifying a checkpoint. The script will list all available models and prompt you to choose one.
    ```bash
    python -m richard.src.test.test --config richard/config/test.yaml
    ```

    **B) Direct Specification:**
    Provide the exact filename of the checkpoint you want to test using the `--ckpt_name` flag.
    ```bash
    python -m richard.src.test.test --config richard/config/test.yaml --ckpt_name "best_checkpoint_xxxxxxxx.pth"
    ```

    *   **Controlling Outputs (Optional):**
        By default, the script only calculates and prints the Dice score. Use these flags to save output files:
        *   `--save_masks`: Saves the final binary prediction masks as `.npy` files.
        *   `--save_overlays`: Saves visual comparison images (`.png`) that overlay the predicted and true masks on the CT scan.

3.  **Analyze the Results:**
    *   **Performance Score:** The primary output is the Dice score report, which is printed to the console and saved in a log file in `richard/logs/`. The log shows the average patient Dice score for each tested threshold and highlights the best-performing one.
    *   **Saved Files:** If you used the save flags, the output files will be in a directory named after the model you tested (e.g., `richard/results/best_checkpoint_xxxxxxxx/`). Inside, you will find the `predicted_masks/` and `overlays/` folders.

## Analyzing Results

To help compare results from multiple training runs, you can use the `analyze_logs.py` script. This tool automatically parses your log files, extracts the key configurations and performance metrics from each run, and compiles them into a single summary file.

*   **How to run it:**
    Execute the following command from the project's root directory:
    ```bash
    python -m richard.src.test.analyze_logs
    ```
*   **What you get:**
    *   A summary table printed directly to your console.
    *   A detailed CSV file named `log_analysis_results.csv` saved in `richard/results` direcory , containing the timestamp, best Dice score, best threshold, and all the key configuration parameters for every training run it finds. This is extremely useful for tracking and comparing your experiments.

Refer to the specific configuration files in `config/` to adjust model hyperparameters, dataset paths, and training settings.