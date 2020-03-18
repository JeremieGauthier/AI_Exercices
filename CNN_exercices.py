#%%
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) Output
        t = self.out(t)
        t = F.softmax(t, dim=1)

        return t

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = tensor.tensor([])

    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
        

if __name__ == "__main__":

    train_set = torchvision.datasets.FashionMNIST(
        # root='train-images-idx3-ubyte',
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )


    network = Network()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    optimizer = optim.Adam(network.parameters(), lr=0.01)

    nb_epochs = 1
    for epoch in range(nb_epochs):
        
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad() # Pass Batch
            loss.backward() # Calculate Gradient
            optimizer.step() # Update weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        
        print("epoch :", epoch, "Total Correct :", total_correct, "loss:", total_loss)


