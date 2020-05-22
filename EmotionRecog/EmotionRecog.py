# facial expression recognization
# created on 21 May, 2020, Fengdan Cui

import torch
import torch.nn as nn
from FaceData import FaceData
from torch.utils.data import DataLoader

# load dataset

batch_size = 20
train_dataset = FaceData('Faces')
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

for img, label in train_loader:
    print(label)

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


# class EmotionRecog():

