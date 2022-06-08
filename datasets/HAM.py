from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class HAM_Dataset(Dataset):
    def __init__(self, x, y) -> None:
        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]

        return data, target