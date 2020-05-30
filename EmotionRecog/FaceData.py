# implement my dataset
# created on 21 May, 2020

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


class FaceData(Dataset):

    # Code the labels
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

        # 'root' is the root directory of all images
        self.root = root

        # 'names' is a list to store the name of images
        self.names = []

        # 'labels' is a list to store the corresponding label of image
        # e.g. 1, 3, 0
        self.labels = []

        # 'dir' is a list to store the folder name, which is also expressions
        # e.g. happy, sad
        self.dir = []
        for folder in os.listdir(self.root):

            # Connect root and sub path to achieve the image path
            path = self.root + '/' + folder
            if os.path.isdir(path):

                # Access to the image
                for file in os.listdir(path):
                    self.names.append(file)
                    self.labels.append(self.code_dic[folder])
                    self.dir.append(folder)

    # According to the given index, return the image data and label
    def __getitem__(self, index):
        path = self.root + '/' + self.dir[index] + '/' + self.names[index]
        face = cv2.imread(path)
        # face = cv2.resize(face, None, fx=0.3, fy=0.3)
        face = cv2.resize(face, (299, 299))

        # Split channels
        b, g, r = cv2.split(face)

        # Histogram equalization on each channel
        b_h = cv2.equalizeHist(b)
        g_h = cv2.equalizeHist(g)
        r_h = cv2.equalizeHist(r)

        # Merge channels
        face_h = cv2.merge((b_h, g_h, r_h))

        # Data normalization
        face_tensor = self.transform(face_h)

        return face_tensor, self.labels[index]

    # return the length of dataset
    def __len__(self):
        return len(self.labels)

