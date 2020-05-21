# implement my dataset
# created on 21 May, 2020

import os
from torch.utils import data
import cv2

class FaceData(data.Dataset):

    def __init__(self, root):
        super(FaceData, self).__init__()
        self.root = root
        self.names = []
        self.labels = []
        for folder in os.listdir(self.root):
            path = self.root + '/' + folder
            if os.path.isdir(path):
                for file in os.listdir(path):
                    self.names.append(file)
                    self.labels.append(folder)
        print(self.names)
        print(self.labels)

    def __getitem__(self, index):

        return self.imgs, self.labels


    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    print('start')
    train_data = FaceData('Faces')

