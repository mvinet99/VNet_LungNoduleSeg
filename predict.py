import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import random
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from dataset import AllDataset
from model import VNet

### Make predictions on val or test set ###

def set_seed(seed: int):
    """Set random number generator seeds for reproducibility.

    This function sets seeds for the random number generators in NumPy, Python's
    built-in random module, and PyTorch to ensure that random operations are
    reproducible. 

    Args:
        seed (int): The seed value to use for setting random number generator seeds.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Set path and device
MY_PATH = '/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/'
DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# Set val / test images path
val_dir = MY_PATH + 'data/val/images/'
pred_dir = MY_PATH + 'data/predicted/masks/'

# Set and load model
model = VNet()
model.load_state_dict(torch.load(MY_PATH+'model.pth',map_location=torch.device(DEVICE)))
model.to(device)

# Run prediction on all images and save to path
model.eval()
with torch.no_grad():

    files = os.listdir(val_dir)
    for file in tqdm(files, 'running prediction...'):
        
        im = np.load(val_dir+file)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im).to(device)
        im = im.unsqueeze(0)
        predMask = model(im)
        predMask = predMask.squeeze(0)
        
        predMask = predMask.cpu().numpy()
        predMask = (predMask > 0.5) * 255

        np.save(pred_dir+file, predMask)

