import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset

device = ("cuda" if torch.cuda.is_available() else "cpu")
path = Path('.')

class digitDataset(Dataset):
    def __init__(self, dset) -> None:
        super().__init__()
        self.ids = dset[:,0]
        self.img = dset[:,1:]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        return torch.asarray(self.img[index]).reshape(1,28,28).permute(0,2,1).type(torch.float).to(device), torch.Tensor([self.ids[index]]).to(device)


def readcsv():
    
    train = pd.read_csv(path / 'train.csv')
    test = pd.read_csv(path / 'test.csv')
    train = train.to_numpy()
    # print(train)
    np.random.shuffle(train)
    # print(train)
    test = test.to_numpy()
    return train, test

def create_dataset():
    train, test = readcsv()
    train_dataset = digitDataset(train[0:37800])
    valid_dataset = digitDataset(train[3780:])
    test_dataset = digitDataset(test)
    return train_dataset, valid_dataset, test_dataset
