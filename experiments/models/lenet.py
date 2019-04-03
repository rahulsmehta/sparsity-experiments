import torch.nn as nn
import matplotlib.pylab as plt
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .plot import plot_training
from .basemodel import BaseModel

class LeNet(BaseModel):
    def __init__(self, trainloader, testloader, device='cuda'):
        super(LeNet, self).__init__(trainloader, testloader, device)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.classifier = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.LogSoftmax(dim=1)
        )
        #self.fc1 = nn.Linear(4*4*50, 500)
        #self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        return x
        #return F.log_softmax(x, dim=1)
