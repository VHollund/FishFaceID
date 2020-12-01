import random

from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms
from torchvision.transforms import transforms

import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

import PIL
from PIL import Image

import numpy as np

class Config():
    # Change part to test different parts of fish
    part = "headR"
    training_dir = "./data2/"+part+"/training/"
    testing_dir = "./data2/"+part+"/testing/"
    train_batch_size = 64
    train_number_epochs = 100
    image_size = 100

def transformation():
    return transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor()
        ])

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# Dataset class
class SiameseNetworkDataset(Dataset):

    def __init__(self,
                 imageFolderDataset: ImageFolder,
                 transform: transforms = None,
                 should_invert: bool = True):

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index: int):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img1_tuple = None
        # Approx. 50% of images should be in same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            # Filter out images of different class and pick random image
            filtered = [x for x in self.imageFolderDataset.imgs if x[1] == img0_tuple[1]]
            img1_tuple = random.choice(filtered)
        else:
            # Filter out images of same class and pick random image
            filtered = [x for x in self.imageFolderDataset.imgs if x[1] != img0_tuple[1]]
            img1_tuple = random.choice(filtered)
        
        # Open image from path in tuple and convert to grayscale ("L")
        img0 = Image.open(img0_tuple[0]).convert("L")
        img1 = Image.open(img1_tuple[0]).convert("L")

        if self.should_invert:
            PIL.ImageOps.invert(img0)
            PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # Use of last parameter is label when training
        label = np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
        # Returns pair of images, and whether they are from the same fish or not
        return img0, img1, torch.from_numpy(label)

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Modules will be added in order they are passed
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # Takes 1 channel and produces 4
            # 1 channel since we transformed to grayscale
            nn.Conv2d(1, 4, kernel_size=3),
            # Relu activation function
            nn.ReLU(inplace=True),
            # Normalize inputs to improve training time
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*Config.image_size*Config.image_size, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # Contrastive loss function from hackernoon article
        # Uses euclidian distance to calculate difference between outputs
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
