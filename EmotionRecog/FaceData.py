# implement my dataset
# created on 21 May, 2020

import os
from torch.utils import data
import cv2

class FaceData(data.Dataset):

    def __init__(self, root):
        super(FaceData, self).__init__()
        self.root = root
        self.imgs = []
        self.labels = []

    def __getitem__(self, index):

        files = os.listdir(self.root)
        print(files)

        return self.imgs, self.labels


    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    print('start')
