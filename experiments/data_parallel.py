# %matplotlib inline
# import matplotlib.pylab as plt
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse

from pruner import * 
from models import *
from stanford_dogs import *

device = 'cuda:0'
BATCH_BASE=64
TRAIN_BATCH_SIZE = BATCH_BASE*torch.cuda.device_count()
TEST_BATCH_SIZE=BATCH_BASE*torch.cuda.device_count()

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = StanfordDOGS(root='./stanford-dogs', train=True, download=True, 
                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                         shuffle=True, num_workers=4)

testset = StanfordDOGS(root='./stanford-dogs', train=False, download=True, 
                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=4)

net = torchvision.models.resnet34(pretrained=True)
net = nn.DataParallel(net)
net = net.to(device)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
N_EPOCH=100
LOG=25

def get_lr(epoch):
    if (epoch+1) >= 75:
        return 1e-3
    elif (epoch+1) >= 50:
        return 5e-3
    return 1e-2

criterion = nn.CrossEntropyLoss()

for epoch in range(N_EPOCH):  # loop over the dataset multiple times
    print('Starting epoch {}'.format(epoch+1))
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=get_lr(epoch), momentum=0.9, weight_decay=1e-4)
    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        # print('Batch {}'.format(batch_idx), flush=True)
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    #Validation
    # print('Calculating validation accuracy')
    net.eval()
    tcorrect = 0.
    running_loss = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # print(outputs.device)
            predictions = outputs.argmax(dim=1)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            bcorrect = labels.eq(predictions).sum()
            tcorrect += bcorrect.item()
    acc = tcorrect/(len(testloader)*testloader.batch_size)
    avg_loss = running_loss/len(testloader)
    print(acc, avg_loss)

Print('Done')