**VNet for Lung Nodule Segmentation**

Explanation of each script:

- LUNA_mask_extraction.py - Extract and create masks for the LUNA16 dataset

- LUNA_sanity_checks.py - Sanity check created LUNA16 masks

- dataset.py - dataset configuration for training the VNet

- evaluate.py - predict masks and evaulate average dice for test / validate set

- model.py - VNet model architecture

- predict.py - predict masks for test / validate set

- preprocessing.py - preprocess LIDC / NLST / UCLA datasets

- train.py - main training script for training the VNet

- utils.py - helper functions for training the VNet
