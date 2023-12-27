import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import model
import datetime, torch, argparse
from model import autoencoderMLP4Layer
from torch import optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from matplotlib import pyplot as plt
import numpy as np

def getIdx():
    print("To select an MNIST image please")
    value = input("Enter a integer between 0 and 5999 and click enter: ")

    return int(value)

# train_transform = transforms.Compose([transforms.ToTensor()])
# train_set = datasets.MNIST ("./data/mnist", train=True, download=True, transform=train_transform)
# plt.imshow(train_set.data[getIdx()], cmap='gray')
# plt.show()

#print(train_set.targets[getIdx()])


# # Is MPS even available? macOS 12.3+
# print(torch.backends.mps.is_available())
#
# # Was the current version of PyTorch built with MPS activated?
# print(torch.backends.mps.is_built())








