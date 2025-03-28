import numpy as np
import nibabel as nib
import numpy as np
import torchio as tio
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

### Check to make sure the LUNA16 masks and images are of correct size and content ###

# Directory where your images are stored
files = os.listdir('F:/LUNA16/Volumes_modified/masks/')

# Loop through each mask in the directory
for file in tqdm(files):

    img = np.load('F:/LUNA16/Volumes_modified/masks/' + file)
    if img.shape != (64,64,64):
        
        print('Wrong shape:', img, img.shape, 'mask')
    
    if not np.any(img == 1):
        print("This image does not contain a value of 1", img, 'mask')


files = os.listdir('F:/LUNA16/Volumes_modified/images/')

# Loop through each image in the directory
for file in tqdm(files):

    img = np.load('F:/LUNA16/Volumes_modified/images/' + file)
    if img.shape != (64,64,64):
        
        print('Wrong shape:', img, img.shape, 'image')
