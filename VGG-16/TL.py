
# https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch import save
import torch


"""""           
Load the data   
"""""
# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

traindir = "/Users/tj/PycharmProjects/CNN_for_EO/data/train_set/images/"
validdir = "/Users/tj/PycharmProjects/CNN_for_EO/data/train_set/mask/"

batch_size = 1

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
}

# Iterate through the dataloader once
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
print(features.shape, labels.shape)

"""""           
Load and prepare] the model 
"""""

# load in the vgg-16
model = models.vgg16(pretrained=True)

# Freeze the model weights
for param in model.parameters():
    param.requires_grad = False


n_inputs = 1   # Number of images
n_classes = 2  # This is a binary classification task

# Add on classifier
model.classifier[-1] = nn.Sequential(
    nn.Linear(in_features=4096, out_features=2),
    nn.LogSoftmax(dim=1)
)


print(model.classifier)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# # Move to gpu
# model = model.to('cuda')
# Distribute across 2 gpus
model = nn.DataParallel(model)

from torch import optim
# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


n_epochs = 10

for epoch in range(n_epochs):
  for data, targets in dataloaders['train']:
    # Generate predictions
    out = model(data)
    # Calculate loss
    loss = criterion(out, targets)
    # Backpropagation
    loss.backward()
    # Update model parameters
    optimizer.step()

for data, targets in dataloaders['train']:
    log_ps = model(data)
    # Convert to probabilities
    ps = torch.exp(log_ps)

