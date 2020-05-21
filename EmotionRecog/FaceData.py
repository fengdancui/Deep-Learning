# implement my dataset
# created on 21 May, 2020

import os
from torch.utils import data
import cv2
import torch

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

    def __getitem__(self, index):
        path = self.root + '/' + self.labels[index] + '/' + self.names[index]
        face = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face)
        return face_tensor, self.labels[index]


    def __len__(self):
        return len(self.labels)

#
# if __name__ == '__main__':
#     print('start')
#     train_data = FaceData('Faces')


