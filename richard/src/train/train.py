import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from richard.src.data.dataset import AllDataset
from richard.src.models.VNet2D import VNet2D
from richard.src.train.trainer import Trainer
from richard.src.utils.utils import setup_logging


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

if __name__ == "__main__":
    train()