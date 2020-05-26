# facial expression recognization
# created on 22 May, 2020, Fengdan Cui

from FaceCNN import FaceCNN
from FaceData import FaceData
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as opt
from sklearn.metrics import classification_report
import numpy as np
import time


class EmotionRecog():

    def __init__(self, root, batch_size, epochs, lr=0.05, weight_decay=1e-8):
        all_data = FaceData(root)
        train_size = int(0.8 * len(all_data))
        test_size = len(all_data) - train_size
        train_data, test_data = data.random_split(all_data, [train_size, test_size])
        train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
        test_loader = data.DataLoader(test_data, test_size, shuffle=True)
        self.train(train_loader, test_loader, epochs, lr, weight_decay)

    def train(self, train_loader, test_loader, epochs, lr, weight_decay):
        cnn_model = FaceCNN()
        loss_fc = nn.CrossEntropyLoss()
        optimizer = opt.SGD(cnn_model.parameters(), lr=lr, weight_decay=weight_decay)

        for e in range(epochs):

            print('Starting the epoch {} / {}'.format(e + 1, epochs))

            all_loss = []
            all_acc = []
            cnn_model.train()
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                out = cnn_model.forward(imgs)
                loss = loss_fc(out, labels)
                all_loss.append(loss.item())
                _, pred = torch.max(out, 1)
                all_acc.append(((pred == labels).sum().item()) / len(labels))
                loss.backward()
                optimizer.step()

            print('After epoch {} / {}, the loss is : {}, the accuracy is : {}'
                  .format(e + 1, epochs, sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc)))

            if (e + 1) % 5 == 0:
                cnn_model.eval()
                cr = self.validate(cnn_model, test_loader, e)
                print('The classification report after {} epoch'.format(e + 1))
                print(cr)

    def validate(self, model, test_loader, e):
        actual = []
        predict = []
        for imgs, labels in test_loader:
            pred = model.forward(imgs)
            pred = np.argmax(pred.data.numpy(), axis=1)
            predict.extend(pred)
            actual.extend(labels.data.numpy())
        cr = classification_report(actual, predict)
        file_w = open('evaluation.txt', 'a')
        localtime = time.asctime(time.localtime(time.time()))
        file_w.write('running after {} epochs at: '.format(e + 1))
        file_w.write(localtime)
        file_w.writelines('\n')
        file_w.write(cr)
        file_w.writelines('\n')

        return cr


if __name__ == '__main__':
    print('start')
    er = EmotionRecog('Faces', 128, 20)
