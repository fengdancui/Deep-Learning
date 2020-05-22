# implement my dataset
# created on 21 May, 2020

import os
from torch.utils.data import Dataset
import cv2
import torch


class FaceData(Dataset):

    code_dic = {
        'Angry': 1,
        'Disgust': 2,
        'Fear': 3,
        'Happy': 4,
        'Neutral': 5,
        'Sad': 6,
        'Surprise': 7
    }

    def __init__(self, root):
        super(FaceData, self).__init__()
        self.root = root
        self.names = []
        self.labels = []
        self.dir = []
        for folder in os.listdir(self.root):
            path = self.root + '/' + folder
            if os.path.isdir(path):
                for file in os.listdir(path):
                    self.names.append(file)
                    self.labels.append(self.code_dic[folder])
                    self.dir.append(folder)

    def __getitem__(self, index):
        path = self.root + '/' + self.dir[index] + '/' + self.names[index]
        face = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, None, fx=0.3, fy=0.3)
        face_tensor = torch.from_numpy(face)
        return face_tensor, self.labels[index]

    def __len__(self):
        return len(self.labels)

# if __name__ == '__main__':
#     print('start')
#     train_data = FaceData('Faces')
