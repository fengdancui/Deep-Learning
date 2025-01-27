# facial expression recognization
# created on 22 May, 2020, Fengdan Cui

from FaceCNN import FaceCNN
from FaceData import FaceData
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as opt
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt


class EmotionRecog():

    # Split dataset and load data
    def __init__(self, root, batch_size, epochs, lr=0.05, weight_decay=1e-8):
        all_data = FaceData(root)
        train_size = int(0.8 * len(all_data))
        test_size = len(all_data) - train_size
        train_data, test_data = data.random_split(all_data, [train_size, test_size])
        train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
        test_loader = data.DataLoader(test_data, test_size, shuffle=True)
        self.train(train_loader, test_loader, epochs, lr, weight_decay)

    # Start to train
    def train(self, train_loader, test_loader, epochs, lr, weight_decay):
        cnn_model = FaceCNN()

        # Initialize loss function and SGD
        loss_fc = nn.CrossEntropyLoss()
        optimizer = opt.SGD(cnn_model.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate decay, when epoch = 20, lr = 0.8*lr
        # when epoch = 51, lr = 0.8*lr again
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50], gamma=0.8)

        # Store data for plotting
        plot_data = {'train_acc': [], 'test_acc': []}

        for e in range(epochs):
            print('Starting epoch {} / {}'.format(e + 1, epochs))

            all_loss = []
            all_acc = []
            cnn_model.train()

            # Load batches
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                # optimizer.
                out = cnn_model.forward(imgs)
                loss = loss_fc(out, labels)
                all_loss.append(loss.item())
                _, pred = torch.max(out, 1)
                all_acc.append(((pred == labels).sum().item()) / len(labels))
                loss.backward()
                # Update parameters
                optimizer.step()

            # Learning rate decay
            scheduler.step()
            avg_loss = sum(all_loss) / len(all_loss)
            avg_acc = sum(all_acc) / len(all_acc)
            plot_data['train_acc'].append(avg_acc)

            # Print loss and accuracy after each epoch
            print('After epoch {} / {}, the training loss is : {}, the accuracy is : {}'
                  .format(e + 1, epochs, avg_loss, avg_acc))

            # Evaluate the model using test data
            if (e + 1) % 5 == 0:
                cnn_model.eval()

                # Classification report and accuracy
                cr, test_acc = self.validate(cnn_model, test_loader)
                plot_data['test_acc'].append(test_acc)

                print('After epoch {}, the classification report is:'.format(e + 1))
                print(cr)

                # Write classification report and accuracy in file
                # record the running time
                file_w = open('evaluation.txt', 'a')
                localtime = time.asctime(time.localtime(time.time()))
                file_w.writelines('running after {} epochs at: {}'.format(e + 1, localtime))
                file_w.writelines('\n')
                file_w.write(cr)
                file_w.writelines('\n')
                file_w.writelines('the test accuracy is {}'.format(test_acc))
                file_w.writelines('\n')
                file_w.close()

        # Plot the model accuracy during training
        plt.plot(np.arange(1, epochs + 1), plot_data['train_acc'], label='train accuracy')
        plt.plot(np.arange(5, epochs + 1, 5), plot_data['test_acc'], label='test accuracy')
        plt.plot(np.arange(1, epochs + 1), np.ones(epochs), 'k')

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Model Accuracy')
        # show a legend on the plot
        plt.legend()
        plt.savefig('model_acc.png')
        # Display a figure.
        plt.show()

    # Evaluate model, input: model to be evaluated and test data
    # return: classification report and accuracy
    def validate(self, model, test_loader):
        actual = []
        predict = []

        for imgs, labels in test_loader:
            pred = model.forward(imgs)

            # Obtain the index of max value
            pred = np.argmax(pred.data.numpy(), axis=1)
            predict.extend(pred)
            actual.extend(labels.data.numpy())
        cr = classification_report(actual, predict)
        acc = accuracy_score(actual, predict)
        return cr, acc


if __name__ == '__main__':
    print('start')
    er = EmotionRecog('Faces', 16, 70)
