# cnn model for processing images
# created on 21 May, 2020, Fengdan Cui

import torch.nn as nn


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1.apply(self.init_weight)
        self.conv2.apply(self.init_weight)
        self.conv3.apply(self.init_weight)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128 * 37 * 37, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    def init_weight(self, model):
        class_name = model.__class__.__name__
        if class_name.find('Conv') != -1:
            model.weight.data.normal_(0.0, 0.04)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out
