import torch
import torch.nn as nn

#creat one hot class

device = ("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolution = nn.Conv2d(1, 3, kernel_size = 3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(676, 512)
        self.bnorm2d = nn.BatchNorm2d(512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.convolution(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.bnorm2d(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

class One_Hot(torch.nn.Module):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).to(device)
    def forward(self, X_in):
        X_in = X_in.long()
        return self.ones.index_select(0,X_in.data)
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)
