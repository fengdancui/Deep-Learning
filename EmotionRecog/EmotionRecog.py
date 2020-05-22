# facial expression recognization
# created on 22 May, 2020, Fengdan Cui

from FaceCNN import FaceCNN
from FaceData import FaceData
from torch.utils.data import DataLoader

class EmotionRecog():

    def __init__(self):
        self.cnn = FaceCNN()

    def train(self):
        batch_size = 20
        train_dataset = FaceData('Faces')
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)