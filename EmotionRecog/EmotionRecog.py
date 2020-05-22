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


# class EmotionRecog():
#
