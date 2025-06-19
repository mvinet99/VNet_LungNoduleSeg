import os
import re
import json
import pandas as pd
from pathlib import Path
import argparse
import logging

# Set up basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_config_from_log(log_content: str) -> dict:
    """Extracts the JSON config block from a training log file."""
    # The config is logged in debug mode after this specific line
    config_marker = "Decorator injected 'cfg' into function call:"
    
    if config_marker not in log_content:
        return {}

    # Find the start of the JSON blob
    json_start_index = log_content.find(config_marker) + len(config_marker)
    json_blob = log_content[json_start_index:].strip()

    # The JSON is pretty-printed, so we need to find its boundaries.
    # It starts with '{' and we need to find the matching '}'.
    open_braces = 0
    json_end_index = 0
    for i, char in enumerate(json_blob):
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
        
        if open_braces == 0 and i > 0:
            json_end_index = i + 1
            break
            
    if json_end_index > 0:
        config_str = json_blob[:json_end_index]
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            logging.warning(f"Could not parse JSON config from log: {e}")
            return {}
    return {}

def flatten_config(config: dict) -> dict:
    """Flattens a nested config dictionary for easier DataFrame creation."""
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                     for ssub_key, ssub_value in sub_value.items():
                         flat_config[f"{key}.{sub_key}.{ssub_key}"] = ssub_value
                else:
                    flat_config[f"{key}.{sub_key}"] = sub_value
        else:
            flat_config[key] = value
            
    # Special handling for criterion to make it readable
    if 'criterion' in config:
        criterion_dict = config['criterion']
        crit_str = ", ".join([f"{name}(w={details.get('weight', 'N/A')})" for name, details in criterion_dict.items()])
        flat_config['criterion_summary'] = crit_str

    return flat_config

def parse_best_dice_from_log(log_content: str) -> tuple:
    """Extracts the best Dice score and threshold from a test log file."""
    # Regex to find the line with the best dice score
    dice_pattern = re.compile(r"Best Average Patient Dice Score: ([\d.]+) at Threshold: ([\d.]+)")
    match = dice_pattern.search(log_content)
    if match:
        dice_score = float(match.group(1))
        threshold = float(match.group(2))
        return dice_score, threshold
    return None, None

def analyze_logs(logs_dir: Path, output_file: Path):
    """
    Parses training and testing logs to extract configurations and results.

    Args:
        logs_dir (Path): The directory containing the log files.
        output_file (Path): The path to save the resulting CSV file.
    """
    if not logs_dir.is_dir():
        logging.error(f"Logs directory not found: {logs_dir}")
        return

    results = []
    
    # Regex to extract timestamp from filenames
    train_log_pattern = re.compile(r"train_(\d{2}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\.log")
    
    logging.info(f"Scanning for log files in {logs_dir}...")
    
    train_logs = sorted(list(logs_dir.glob("train_*.log")), reverse=True)
    
    for train_log_path in train_logs:
        match = train_log_pattern.match(train_log_path.name)
        if not match:
            continue
            
        timestamp = match.group(1)
        logging.info(f"Processing run with timestamp: {timestamp}")
        
        # Find the corresponding test log. We check for common patterns.
        test_log_final_path = logs_dir / f"test_final_checkpoint_{timestamp}.log"
        test_log_best_path = list(logs_dir.glob(f"test_best_checkpoint_{timestamp}*.log"))

        test_log_path = None
        if test_log_final_path.exists():
            test_log_path = test_log_final_path
        elif test_log_best_path:
            test_log_path = test_log_best_path[0] # Take the first match if multiple exist

        if not test_log_path:
            logging.warning(f"No matching test log found for train log: {train_log_path.name}")
            continue

        # --- Parse Test Log ---
        test_log_content = test_log_path.read_text(encoding='utf-8', errors='ignore')
        best_dice, best_threshold = parse_best_dice_from_log(test_log_content)
        
        if best_dice is None:
            logging.warning(f"Could not find best Dice score in {test_log_path.name}")
            continue

        # --- Parse Train Log for Config ---
        train_log_content = train_log_path.read_text(encoding='utf-8', errors='ignore')
        config = parse_config_from_log(train_log_content)
        
        run_data = {
            'timestamp': timestamp,
            'best_dice': best_dice,
            'best_threshold': best_threshold,
        }
        
        if config:
            flat_config = flatten_config(config)
            run_data.update(flat_config)
            
        results.append(run_data)

    if not results:
        logging.info("No matching log pairs found or no data could be extracted.")
        return

    # --- Create and Save DataFrame ---
    df = pd.DataFrame(results)
    
    # Reorder columns to be more readable
    core_cols = ['timestamp', 'best_dice', 'best_threshold', 'criterion_summary', 'model', 'optimizer.name', 'optimizer.lr']
    # Get remaining columns and sort them for consistency
    other_cols = sorted([col for col in df.columns if col not in core_cols])
    
    # Ensure all core columns exist before trying to reorder
    final_cols = [col for col in core_cols if col in df.columns] + other_cols
    df = df[final_cols]

    # Save to CSV
    df.to_csv(output_file, index=False)
    
    logging.info(f"Analysis complete. Results saved to: {output_file}")
    
    # Print a summary to console
    print("\n--- Log Analysis Summary ---")
    print(df[['timestamp', 'best_dice', 'best_threshold', 'criterion_summary', 'optimizer.lr']].to_string())
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training and testing logs to extract results and configurations.")
    parser.add_argument("--logs_dir", type=str, default="richard/logs",
                        help="Path to the directory containing the log files.")
    parser.add_argument("--output_file", type=str, default="richard/logs/log_analysis_results.csv",
                        help="Path to save the output CSV file.")
    
    args = parser.parse_args()
    
    logs_path = Path(args.logs_dir)
    output_path = Path(args.output_file)
    
    analyze_logs(logs_path, output_path)