import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from dataset import AllDataset
from model import VNet
#from utils import DiameterLoss

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

MY_PATH = '/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/LUNA16/'

# Filepaths to train, validate masks and images
train_dir = MY_PATH + 'data/train/images/'
train_maskdir = MY_PATH + 'data/train/masks/'
#train_diamdir = MY_PATH + 'data/train/diam/'
val_dir = MY_PATH + 'data/val/images/'
val_maskdir = MY_PATH + 'data/val/masks/'
#val_diamdir = MY_PATH + 'data/val/diam/'
pred_dir = MY_PATH + 'data/predicted/'

### Set model and hyperparameters ###
DEVICE = "cuda:1"
EPOCHS = 50
batch_size = 8
model = VNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Create training and validation datasets
train_dataset = AllDataset(image_dir=train_dir,
                            mask_dir=train_maskdir)
                            #diam_dir=train_diamdir)
val_dataset = AllDataset(image_dir=val_dir,
                            mask_dir=val_maskdir)
                            #diam_dir=val_diamdir)

# Create DataLoader for training and validation sets
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define function for training epoch
def train_one_epoch(device):

    running_loss = 0

    # Run for each batch in the training loader
    for i, data in tqdm(enumerate(training_loader), total=len(training_loader), desc=f"Training Epoch {epoch + 1}"):
        
        # Load data & label
        inputs, labels = data #, diameters
        
        # Cast data to GPU
        inputs, labels = inputs.to(device), labels.to(device) #, diameters.to(device) , diameters

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions for current batch
        outputs = model(inputs)

        # Compute loss and backprop gradients
        loss = loss_fn(outputs, labels.long())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Report loss
        running_loss += loss.item()

    return running_loss/(i+1)

# Set GPU runtime
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print("Training on", device)

# Move model to the GPU
model.to(device)

# Define training loop
epoch_number = 0
training_losses = []
validation_losses = []
for epoch in range(EPOCHS):
    print('Epoch {}'.format(epoch_number + 1))

    # Turn on gradient tracking
    model.train(True)
    avg_loss = train_one_epoch(device)

    running_vloss = 0.0

    # Set to evaluation mode
    model.eval()

    # Validation - disable gradient tracking
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader), desc=f"Validation Epoch {epoch + 1}"):

            vinputs, vlabels = vdata # , vdiams

            # Cast data to GPU
            vinputs, vlabels= vinputs.to(device), vlabels.to(device) # , vdiams , vdiams.to(device) 

            # Make predictions on validate batch
            voutputs = model(vinputs)

            # Caluclate and report loss
            vloss = loss_fn(voutputs, vlabels.long())
            running_vloss += vloss            
    avg_vloss = running_vloss / (i + 1)

    # Display train & validate loss
    print('Loss is: train {} val {}'.format(avg_loss, avg_vloss))

    training_losses.append(avg_loss)
    validation_losses.append(avg_vloss)

    epoch_number += 1

model_path = MY_PATH + 'model.pth'
torch.save(model.state_dict(), model_path)

# Plotting and saving the loss curve after training finishes
plt.figure(figsize=(10,5))
plt.plot(training_losses, label='Training Loss')
validation_losses = torch.tensor(validation_losses).cpu()
validation_losses = validation_losses.numpy(force=True)
plt.plot(validation_losses, label='Validation Loss')
plt.title('Loss Curve - Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the loss curve to a file
plt.savefig(MY_PATH + 'Loss.png')

print('Finished')
