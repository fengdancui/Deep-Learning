# facial expression recognization
# created on 22 May, 2020, Fengdan Cui

from FaceCNN import FaceCNN
from FaceData import FaceData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class EmotionRecog():

    # def __init__(self):
    # self.cnn = FaceCNN()

    def train(self, train_data, test_data, batch_size, epochs, learning_rate):
        cnn_model = FaceCNN()
        batch_size = 20
        # train_dataset = FaceData('Faces')
        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        loss_fc = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate)

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

