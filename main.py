import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from img_convert import create_dataset
from model import One_Hot, CNN
from tqdm import tqdm
from pathlib import Path
import csv

f = open('pred.txt', 'w')

device = ("cuda" if torch.cuda.is_available() else "cpu")
path = Path('.')
digitCNN = CNN().to(device)
CEloss = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(digitCNN.parameters(), lr = 0.0001)
train_set, valid_set, test_set = create_dataset()
batch = 1024
one_hot = One_Hot(10).to(device)

num_epochs = 25
valid_every = 5
def train():
    train_loader = DataLoader(train_set, batch_size = batch)
    valid_loader = DataLoader(valid_set, batch_size = batch)
    for epoch in tqdm(range(1, num_epochs + 1)):
        for idx, item in enumerate(train_loader):
            output = digitCNN(item[0])
            target = item[1]
            target = one_hot(torch.squeeze(target))
            loss = CEloss(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if(epoch % valid_every == 0):
            correct = 0
            for id, item in enumerate(valid_loader):
                output = digitCNN(item[0])
                pred = output.data.max(1, keepdim=True)[1]
                target = item[1]
                correct += (pred == target).sum()
            print("Accuracy = ", correct.item()/valid_set.__len__())

def test():
    test_loader = DataLoader(test_set, batch_size = batch)
    const = 1
    f.write("ImageId,Label \n")
    m = lambda z, const: [f.write(str(i + const) + ',' + str(z[i].item()) + '\n') for i in range(len(z))]
    for idx, item in tqdm(enumerate(test_loader)):
        output = digitCNN(item)
        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.cpu().numpy()
        m(pred, const)
        const += 1024
        


train()
test()


