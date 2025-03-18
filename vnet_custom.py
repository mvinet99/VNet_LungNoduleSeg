# Import modules
import numpy as np
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from dataset import AllDataset
from model import VNet
from utils import DiameterLoss
MY_PATH = '/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/'
# Parameters for dataset
train_dir = MY_PATH + 'data/splits/train/images/'
train_maskdir = MY_PATH + 'data/splits/train/masks/'
train_diamdir = MY_PATH + 'data/splits/train/diam/'
val_dir = MY_PATH + 'data/splits/val/images/'
val_maskdir = MY_PATH + 'data/splits/val/masks/'
val_diamdir = MY_PATH + 'data/splits/val/diam/'
pred_dir = MY_PATH + 'data/splits/predicted/'

batch_size = 16   # Batch size for the DataLoader

# Create training and validation datasets
train_dataset = AllDataset(image_dir=train_dir,
                            mask_dir=train_maskdir,
                            diam_dir=train_diamdir)
val_dataset = AllDataset(image_dir=val_dir,
                            mask_dir=val_maskdir,
                            diam_dir=val_diamdir)

#train_dataset = ZeroDataset(num_samples=15)
#val_dataset = ZeroDataset(num_samples=15)

# Create DataLoader for training and validation sets
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = VNet()

# Define loss function and optimizer 

loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = DiameterLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define function for training epoch

def train_one_epoch(epoch_index, tb_writer, device):
    running_loss = 0.
    last_loss = 0.

    # Run for each batch in the training loader
    for i, data in tqdm(enumerate(training_loader), total=len(training_loader), desc=f"Training Epoch {epoch + 1}"):
        
        # Load data & label
        inputs, labels, diameters = data
        
        # Cast data to GPU
        inputs, labels, diameters = inputs.to(device), labels.to(device), diameters.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions for current batch
        outputs = model(inputs)

        outputs = outputs.view(inputs.shape[0], 2, 96, 96, 32)

        # Compute loss and gradients
        loss = loss_fn(outputs, labels.long())
        #loss = loss_fn(outputs, diameters)

        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Report loss to data writer
        running_loss += loss.item()

    return running_loss/(i+1)

# Set GPU runtime
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print("Training on", device)

# Move model to the GPU
model.to(device)

# Define per-epoch training loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer  = SummaryWriter('runs/testing_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 20

best_vloss = 1000000.

training_losses = []
validation_losses = []
for epoch in range(EPOCHS):
    print('Epoch {}'.format(epoch_number + 1))

    # Turn on gradient tracking
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, device)

    running_vloss = 0.0

    # Set to evaluation mode
    model.eval()

    # Disable gradient tracking
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader), desc=f"Validation Epoch {epoch + 1}"):
            vinputs, vlabels, vdiams = vdata
            # Cast data to GPU
            vinputs, vlabels, vdiams = vinputs.to(device), vlabels.to(device), vdiams.to(device)
            voutputs = model(vinputs)
            voutputs = voutputs.view(vinputs.shape[0], 2, 96, 96, 32)
            vloss = loss_fn(voutputs, vlabels.long())
            running_vloss += vloss
                
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    training_losses.append(avg_loss)
    validation_losses.append(avg_vloss)


    # if (epoch_number+1) == EPOCHS:
    #     # Set to evaluation mode
    #     model.eval()
    #     with torch.no_grad():

    #         files = os.listdir(val_dir)
    #         for file in files:
                
    #             im = np.load(val_dir+file)
    #             im = np.expand_dims(im, 0)
    #             im = torch.from_numpy(im).to(device)
    #             im = im.unsqueeze(0)
    #             predMask = model(im)
    #             predMask = predMask.view(1, 2, 96, 96, 32)
    #             predMask = predMask.squeeze(0)
                
    #             predMask = predMask.cpu().numpy()
    #             predMask = (predMask > 0.5) * 255

    #             np.save(pred_dir+file, predMask)

    # Log the loss averaged per batch for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number+1)
    
    writer.flush()

    # # Track the best performance of the model & save state
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = MY_PATH + 'custom_vnet/'+'model_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)

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
