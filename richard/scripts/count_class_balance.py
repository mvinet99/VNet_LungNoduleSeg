#!/usr/bin/env python3
"""
Count class balance (zeros vs ones) in mask files, verifying masks/images file counts.
"""
import argparse
import logging
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count zeros and ones in mask directory for class imbalance analysis."
    )
    parser.add_argument(
        '--image_dir', type=Path, default="/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/data/splits/train/images_2D_0axis",
        help='Directory containing input image files (.npy).'
    )
    parser.add_argument(
        '--mask_dir', type=Path, default="/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/data/splits/train/masks_2D_0axis",
        help='Directory containing mask files (.npy) with binary labels (0 or 1).'  
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir

    if not image_dir.exists() or not image_dir.is_dir():
        logging.error(f"Image directory not found: {image_dir}")
        return
    if not mask_dir.exists() or not mask_dir.is_dir():
        logging.error(f"Mask directory not found: {mask_dir}")
        return

    # Collect .npy files
    image_files = sorted(image_dir.glob('*.npy'))
    mask_files = sorted(mask_dir.glob('*.npy'))

    logging.info(f"Found {len(image_files)} image files and {len(mask_files)} mask files.")
    if len(image_files) != len(mask_files):
        logging.warning("Number of images and masks differ. Proceeding with mask files only.")

    zero_count = 0
    one_count = 0
    processed = 0

    for mask_path in mask_files:
        try:
            mask = np.load(mask_path)
        except Exception as e:
            logging.warning(f"Skipping non-numpy mask file: {mask_path}")
            continue
        zeros = int((mask == 0).sum())
        ones = int((mask == 1).sum())
        zero_count += zeros
        one_count += ones
        processed += 1

    if processed == 0:
        logging.error("No valid mask files processed.")
        return

    total = zero_count + one_count
    neg_ratio = zero_count / total if total > 0 else 0
    pos_ratio = one_count / total if total > 0 else 0

    print(f"Processed {processed} mask files.")
    print(f"Total pixels: {total}")
    print(f"  Zeros: {zero_count:,} ({neg_ratio:.4f})")
    print(f"  Ones:  {one_count:,} ({pos_ratio:.4f})")
    if one_count > 0:
        print(f"Suggested pos_weight (neg/pos): {zero_count/one_count:.4f}")
    else:
        print("No positive pixels found; cannot compute pos_weight.")

if __name__ == '__main__':
    main() 