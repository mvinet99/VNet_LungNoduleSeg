import numpy as np
import os

# Define target shape
target_shape = (96, 96, 16)

# Directory containing the numpy files
directory = '/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/NLST/labels/'

# List all files in the directory
files = os.listdir(directory)

for file in files:
    file_path = os.path.join(directory, file)
    
    # Load the numpy array
    n = np.load(file_path)
    
    # Check if the shape is incorrect
    if n.shape != target_shape:
        print(f"Modifying {file} from shape {n.shape} to {target_shape}")
        
        # Handle cases where the third dimension is larger than target
        if n.shape[2] > target_shape[2]:
            n = n[:, :, :target_shape[2]]  # Crop along the third dimension
        
        # Calculate padding for each dimension
        pad_width = [(0, max(0, target_shape[i] - n.shape[i])) for i in range(3)]
        
        # Apply zero-padding
        n_padded = np.pad(n, pad_width, mode='constant', constant_values=0)
        
        # Save the modified array, overwriting the original file
        np.save(file_path, n_padded)
        print(f"Saved modified array to {file_path}")
