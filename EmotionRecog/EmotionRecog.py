# facial expression recognization
# created on 22 May, 2020, Fengdan Cui

from FaceCNN import FaceCNN
from FaceData import FaceData
from torch.utils import data
import torch.nn as nn
import torch.optim as opt


class EmotionRecog():

    def __init__(self, root, batch_size, epochs, lr):
        all_data = FaceData(root)
        train_size = int(0.8 * len(all_data))
        test_size = len(all_data) - train_size
        train_data, test_data = data.random_split(all_data, [train_size, test_size])
        train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_data, batch_size, shuffle=True)
        self.train(train_loader, epochs, lr)

    def train(self, train_loader, epochs, lr):
        cnn_model = FaceCNN()
        loss_fc = nn.CrossEntropyLoss()
        optimizer = opt.SGD(cnn_model.parameters(), lr=lr)

        for e in range(epochs):
            loss = 0
            cnn_model.train()
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                out = cnn_model.forward(imgs)
                loss = loss_fc(out, labels)
                loss.backward()
                optimizer.step()

            print('After {} epochs , the loss_rate is : '.format(e + 1), loss.item())


if __name__ == '__main__':
    print('start')
    er = EmotionRecog('Faces', 10, 20, 0.05)
