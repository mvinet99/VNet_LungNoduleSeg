import os
import re
import pandas as pd
from pathlib import Path
import logging
import argparse

# Configure basic logging for the analysis script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_file(log_path: Path) -> pd.DataFrame:
    """Parses a single debug log file to extract epoch metrics."""
    epoch_data = []
    current_epoch = None
    epoch_metrics = {}

    # Regular expressions to find metric lines
    epoch_re = re.compile(r"Epoch (\d+) Summary:")
    train_re = re.compile(r"Train Loss: ([\d.]+), Train Dice: ([\d.]+)")
    val_re = re.compile(r"Val Loss: ([\d.]+),\s+Val Dice: ([\d.]+)") # Allow potential extra spaces
    best_re = re.compile(r"Best Val Dice So Far: ([\d.]+)")

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Match epoch start
                epoch_match = epoch_re.search(line)
                if epoch_match:
                    # If we have data from a previous epoch, store it
                    if current_epoch is not None and epoch_metrics:
                        epoch_metrics['epoch'] = current_epoch
                        epoch_data.append(epoch_metrics)
                    
                    # Start new epoch
                    current_epoch = int(epoch_match.group(1))
                    epoch_metrics = {} # Reset metrics for the new epoch
                    continue # Move to next line after finding epoch start

                if current_epoch is None: # Skip lines before the first epoch summary starts
                    continue

                # Match metric lines only after an epoch has started
                train_match = train_re.search(line)
                if train_match:
                    epoch_metrics['train_loss'] = float(train_match.group(1))
                    epoch_metrics['train_dice'] = float(train_match.group(2))
                    continue

                val_match = val_re.search(line)
                if val_match:
                    epoch_metrics['val_loss'] = float(val_match.group(1))
                    epoch_metrics['val_dice'] = float(val_match.group(2))
                    continue
                    
                best_match = best_re.search(line)
                if best_match:
                    epoch_metrics['best_val_dice_so_far'] = float(best_match.group(1))
                    # Note: We store this, but might not use it directly in summary,
                    # focusing instead on the actual val_dice of the best epoch.
                    continue 

            # Append the last epoch's data after the loop finishes
            if current_epoch is not None and epoch_metrics:
                 epoch_metrics['epoch'] = current_epoch
                 epoch_data.append(epoch_metrics)

    except FileNotFoundError:
        logging.error(f"Log file not found: {log_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error parsing log file {log_path}: {e}")
        return pd.DataFrame()
        
    if not epoch_data:
         logging.warning(f"No epoch summary data found in {log_path}")
         return pd.DataFrame()

    df = pd.DataFrame(epoch_data)
    # Ensure essential columns exist, even if some lines were missed in the log
    for col in ['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice']:
        if col not in df.columns:
            df[col] = pd.NA 
    
    # Reorder columns for clarity
    cols_order = ['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'best_val_dice_so_far']
    # Include best_val_dice_so_far only if it was actually found in logs
    if 'best_val_dice_so_far' not in df.columns:
       cols_order.pop() 
    df = df[cols_order]
    
    return df.sort_values(by='epoch').reset_index(drop=True)


def analyze_logs(log_dir: Path, results_dir: Path):
    """Finds log files, parses them, saves detailed CSVs and a summary CSV."""
    
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved in: {results_dir.resolve()}")

    log_files = list(log_dir.glob("debug_*.log"))
    if not log_files:
        logging.warning(f"No 'debug_*.log' files found in {log_dir.resolve()}")
        return

    logging.info(f"Found {len(log_files)} log files to process.")
    
    all_summaries = []

    for log_path in log_files:
        run_timestamp = log_path.stem.replace("debug_", "") # Extract timestamp identifier
        logging.info(f"Processing: {log_path.name} (Run ID: {run_timestamp})")

        df_run = parse_log_file(log_path)

        if df_run.empty:
            logging.warning(f"Skipping {log_path.name} due to parsing issues or no data.")
            continue
            
        # Save detailed results for this run
        detailed_csv_path = results_dir / f"{run_timestamp}_metrics.csv"
        df_run.to_csv(detailed_csv_path, index=False)
        logging.info(f"Saved detailed metrics to: {detailed_csv_path.name}")

        # --- Generate Summary ---
        # Find the epoch with the best validation dice score
        best_epoch_idx = df_run['val_dice'].idxmax() 
        if pd.isna(best_epoch_idx):
             logging.warning(f"Could not determine best epoch for run {run_timestamp} (maybe no val_dice?). Skipping summary.")
             continue
             
        best_epoch_data = df_run.loc[best_epoch_idx]

        summary = {
            'run_timestamp': run_timestamp,
            'best_epoch': int(best_epoch_data['epoch']),
            'best_val_dice': best_epoch_data['val_dice'],
            'val_loss_at_best': best_epoch_data['val_loss'],
            'train_loss_at_best': best_epoch_data['train_loss'],
            'train_dice_at_best': best_epoch_data['train_dice'],
            'total_epochs': df_run['epoch'].max()
        }
        all_summaries.append(summary)
        # -----------------------

    if not all_summaries:
        logging.warning("No valid summary data generated from any log file.")
        return

    # Save the overall summary
    df_summary = pd.DataFrame(all_summaries)
    summary_csv_path = results_dir / "summary.csv"
    df_summary.sort_values(by='run_timestamp', inplace=True) # Sort by run time
    df_summary.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved summary of all runs to: {summary_csv_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training logs and generate result summaries.")
    # Use relative paths from the workspace root as defaults
    parser.add_argument("--log_dir", type=str, default="richard/logs",
                        help="Directory containing the 'debug_*.log' files.")
    parser.add_argument("--results_dir", type=str, default="richard/results",
                        help="Directory where analysis results (CSVs) will be saved.")
    
    args = parser.parse_args()

    # Resolve paths relative to the script's assumed location or workspace root
    # This assumes the script is run from the workspace root or paths are relative to it
    workspace_root = Path().cwd() # Or specify a fixed root if needed
    log_directory = workspace_root / args.log_dir
    results_directory = workspace_root / args.results_dir

    analyze_logs(log_dir=log_directory, results_dir=results_directory)
    logging.info("Analysis complete.") 