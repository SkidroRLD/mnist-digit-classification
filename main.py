import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from img_convert import create_dataset
from model import One_Hot, CNN
from tqdm import tqdm


digitCNN = CNN()
CEloss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(digitCNN.parameters(), lr = 0.0001)
train_set, test_set = create_dataset()
batch = 256
one_hot = One_Hot()



def train():
    train_loader = DataLoader(train_set, batch_size = batch)
    for idx, item in enumerate(train_loader):
        output = digitCNN(item[0])
        target = item[1]
        target = one_hot(target)
        loss = CEloss(output, target)
        optim.zero_grad()
        loss.backward()
        optim.step()

def test():
    test_loader = DataLoader(test_set, batch_size = 1)
    correct_pred = 0
    for idx, item in enumerate(test_loader):
        output = digitCNN(item[0])
        pred = output.data.max(1, keepdim=True)[1]
        target = item[1]
        correct += (pred == target)


        


