import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import Dataset
import torch.utils.data as data
import glob
import os
from PIL import Image
from libtiff import TIFF

from PIL import Image
import numpy as np

from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

im = Image.open('/Users/tj/Desktop/Screenshot 2020-06-12 at 13.39.24.png')
print(type(im))
im = np.asarray(im)
im = im[:, :, 0:3]


min_img_size = 224
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
inp = transform_pipeline(im).unsqueeze(0)
print(inp)

# Pass the input through the net
out = fcn(inp)['out']
print (out.shape)



# Confirm access to a cpu or a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Load in the data

trainset = "data/test_set/images"
testset = "data/train_set"

print(len(testset))
#
# print(len(trainloader))
#
# for images, labels in trainloader:
#     print(images.size(), labels.size())
#     break
#
# # Ask PyTorch for the pre-trained VGG-16 model
# vgg16 = models.vgg16(pretrained=True)
#
# # Set optimiser to Adam
# optimiser = Adam(vgg16.parameters())
#
# print(vgg16.features)
# print(vgg16.classifier)
#
# # When treating the network as a fixed feature extractor, we need to freeze the network
# # Grab all the parameters, and set the requires grad to false.
# for param in vgg16.parameters():
#     param.requires_grad = False
#
# # Then remove the models last fully connected layer, and use the fixed feature extractor
# # THen add a linear classifier.
# # Set the classes to 2
# vgg16.classifier[-1] = nn.Sequential(
#     nn.Linear(in_features=4096, out_features=2),
#     nn.LogSoftmax(dim=1)
# )
#
# # Check again the architecture of the network
# print(vgg16.features)
# print(vgg16.classifier)
#
# # Change the criterion to Sigmoid, for binary classification tasks
# criterion = nn.Sigmoid()
#
# num_epochs = 1   # Define the number of iterations
# batch_loss = 0
# cum_epoch_loss = 0  # Sets a tracker for loss
#
#
#
#
#
#
# """""
# Running data through the network
# """""
#
# model = vgg16.to(device)
# optimiser = Adam(vgg16.parameters(()))
#
# for e in range(num_epochs):    # for each predefined epoch
#   cum_epoch_loss = 0
#   for batch, (images, labels) in enumerate(trainloader, 1):
#     images = images.to(device)  # Load the images to the device
#     labels = labels.to(device)  # Load the mask to the device
#
#     optimiser.zero_grad() # Zero out the gradients at each iter
#     logps = vgg16(images)  # Run the batch through the VGG-16 model, to see what predictions the model will provide
#     loss = criterion(logps, labels) # Calculate the loss
#     loss.backward() # Run the backwards pass
#     optimiser.step() # Update the weights using the loss values
#
#     batch_loss += loss.item()
#     print(f"Epoch({e}/{num_epochs} : number ({batch}/{len(trainloader)}))) Batch loss : {loss.item()}")
#
#     print(f"Training loss : {batch_loss/len(trainloader)}")
#
#
# """
# Evaluate the mode
# During training, the output layer randomly sets some of its inputs to zero which effectivley erases them from the
# network
# This makes the finley trained network more robust
#
# """
#
# vgg16.to("cpu")
# vgg16.eval()
#
# with torch.no_grad():
#     images, labels = next(iter(testloader))
#     logps = vgg16(images)
#
#     output = torch.exp(logps)
#     print(output) # Prints the probabilities
#
#     pred = torch.argmax(output, 1)
#
#
#
#

