# implement my dataset
# created on 21 May, 2020

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import torch


class FaceData(Dataset):

    code_dic = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Neutral': 4,
        'Sad': 5,
        'Surprise': 6
    }

    # range [0, 255] -> [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

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
        face = cv2.imread(path)
        # face = cv2.resize(face, None, fx=0.3, fy=0.3)
        face = cv2.resize(face, (299, 299))

        b, g, r = cv2.split(face)
        b_h = cv2.equalizeHist(b)
        g_h = cv2.equalizeHist(g)
        r_h = cv2.equalizeHist(r)
        face_h = cv2.merge((b_h, g_h, r_h))
        face_tensor = self.transform(face_h)
        # face_tensor = face_tensor.permute(2, 0, 1)
        # face_tensor = torch.from_numpy(face_h).float()
        # face_tensor = self.transform(face_h)
        return face_tensor, self.labels[index]

    def __len__(self):
        return len(self.labels)

