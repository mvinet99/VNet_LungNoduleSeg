import pandas as pd
import matplotlib.pyplot as plt
import torchio as tio
import nibabel as nib
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import json

val = True

IMAGES_PATH = f"/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/splits/{'val' if val else 'train'}/images"
MASKS_PATH = f"/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/splits/{'val' if val else 'train'}/masks"

images_files = os.listdir(IMAGES_PATH)
masks_files = os.listdir(MASKS_PATH)

images_files = sorted(images_files, key=lambda x: int(x[4:].replace('.npy', '')))
masks_files = sorted(masks_files, key=lambda x: int(x[4:].replace('.npy', '')))

lidc_files = [filename for filename in images_files if filename.startswith('LIDC')]
print(len(lidc_files))

nlst_files = [filename for filename in images_files if filename.startswith('NLST')]
print(len(nlst_files))

ucla_files = [filename for filename in images_files if filename.startswith('UCLA')]
print(len(ucla_files))

def load_npy(dirpath, idx):
    filenames:list[str] = os.listdir(dirpath)
    filenames = sorted(filenames, key=lambda x: int(x[4:].replace('.npy', '')))
    filename:str = filenames[idx]
    filepath:str = os.path.join(dirpath, filename)
    return np.load(filepath), filename

patient_idxs = np.arange(len(images_files))
axis_to_slice = 0
middle_slice_idx = 32

# --- Part 1: Iterative loading and counting (necessary due to separate files) ---
print("Loading masks and counting positive voxels...")
pos_counts_list = []
mask_filenames = np.array(masks_files) # Get sorted filenames once

for patient_idx in patient_idxs:
    # Load the specific mask file directly instead of listing/sorting in load_npy
    filepath = os.path.join(MASKS_PATH, mask_filenames[patient_idx])
    mask_array = np.load(filepath)
    # mask_array, _ = load_npy(MASKS_PATH, patient_idx) # Original incorrect line
    pos_count = np.sum(mask_array == 1)
    pos_counts_list.append(pos_count)
pos_counts_array = np.array(pos_counts_list)

# --- Part 2: Vectorized duplicate detection ---
print("\nDetecting duplicate groups (vectorized)...")

# Find indices where the count changes
diffs = np.diff(pos_counts_array)
change_indices = np.where(diffs != 0)[0]

# Determine start and end indices of each sequence
starts = np.insert(change_indices + 1, 0, 0)
ends = np.append(change_indices, len(pos_counts_array) - 1)

# Calculate lengths of sequences
lengths = ends - starts + 1

# Filter for sequences longer than 1 (duplicates)
is_duplicate_group = lengths > 1
duplicate_starts = starts[is_duplicate_group]
duplicate_ends = ends[is_duplicate_group]
duplicate_counts = pos_counts_array[duplicate_starts]

# Construct the final list in the desired format
duplicated_patient_idxs_vectorized = []
for count, start, end in zip(duplicate_counts, duplicate_starts, duplicate_ends):
    indices = list(np.arange(start, end + 1))
    duplicated_patient_idxs_vectorized.append((count, indices))

# --- Output Results ---
print(f"\nNumber of duplicated patient sets: {len(duplicated_patient_idxs_vectorized)}")
total_duplicated_counts = sum(len(indices) for _, indices in duplicated_patient_idxs_vectorized)
print(f"Total duplicated counts: {total_duplicated_counts - len(duplicated_patient_idxs_vectorized)}") # Subtract count of sets themselves
print(duplicated_patient_idxs_vectorized)

duplicated_filenames_vectorized = []
image_filenames = np.array(images_files)

for count, indices in duplicated_patient_idxs_vectorized:
    duplicated_filenames_vectorized.append((count, list(image_filenames[indices])))
for i in range(len(duplicated_filenames_vectorized)):
    print(duplicated_filenames_vectorized[i])












