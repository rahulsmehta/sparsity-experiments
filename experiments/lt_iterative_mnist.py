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

from pruner import SparsityPruner 
from models import *


import argparse

parser = argparse.ArgumentParser(description='Run an experiment.')
parser.add_argument('name', metavar='EXPERIMENT_NAME', type=str,
                    help='A name for the experiment')
parser.add_argument('--epoch', metavar='E', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--prune', metavar='P', type=int, default=10,
                    help='Number of pruning iterations')


args = parser.parse_args()


EXPERIMENT_NAME = args.name
N_EPOCH = args.epoch
pruning_iter = args.prune

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load train and test set
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 512

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.MNIST(root='./mnist-data', train=True,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                         shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='./mnist-data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=1)

Network = LeNet
print('Using {}'.format(Network))

# Train base network
print('Training base network...')
net_base = Network(trainloader, testloader)
net_base = net_base.to(device)
torch.save(net_base.state_dict(), './checkpoints/iterative-pruning/{}-init'.format(EXPERIMENT_NAME))

optimizer = optim.Adam(net_base.parameters(), lr=1e-3, weight_decay=5e-4)
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(N_EPOCH):
    print('Starting epoch {}'.format(epoch+1))
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    train_losses, val_losses, train_accs, val_accs = net_base.train_epoch(epoch, optimizer, plot=False, data=plt_data, LOG=10)

torch.save(net_base.state_dict(), './checkpoints/iterative-pruning/{}-trained'.format(EXPERIMENT_NAME))
save_data = {'train_losses': train_losses, 
             'val_losses': val_losses, 
             'train_accs': train_accs, 
             'val_accs': val_accs}
pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning/{}-init.csv'.format(EXPERIMENT_NAME), index=None)

if device == 'cuda':
    torch.cuda.empty_cache() 
    
# One-shot pruning & fine-tuning
to_retain = 0.2
#pruning_iter = 5
to_retain_iter = to_retain**(1/pruning_iter)

net_ft = Network(trainloader, testloader)
net_ft = net_ft.to(device)
net_ft.load_state_dict(torch.load('./checkpoints/iterative-pruning/{}-trained'.format(EXPERIMENT_NAME)))
val_acc, _ = net_ft.test()
before_count = net_ft.param_count()

print('Before pruning: {}, params: {}'.format(val_acc, before_count))
pruner = SparsityPruner(net_ft)

optimizer = optim.SGD(net_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for prune_epoch in range(pruning_iter):
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    pruner.prune(to_retain_iter)
    after_count = net_ft.param_count()
    print('Starting pruning iteration {}, % pruned: {}'.format(prune_epoch+1, after_count/before_count))
    train_losses, val_losses, train_accs, val_accs = net_ft.train_epoch(prune_epoch, optimizer, plot=False, data=plt_data, LOG=10, 
                                                                    pruner=pruner, early_stop=val_acc)

print('After pruning: {}, params: {}'.format(net_ft.test()[0], after_count))

pruner_path = "experiment_data/iterative-pruning/{}.p".format(EXPERIMENT_NAME)
pickle.dump(pruner, open(pruner_path, "wb" ) )
    

# Retrain from winning ticket initialization
print('Retraining from winning ticket initialization...')
net_retrain = Network(trainloader, testloader)
net_retrain = net_retrain.to(device)
net_retrain.load_state_dict(torch.load('./checkpoints/iterative-pruning/{}-init'.format(EXPERIMENT_NAME)))

_masks = pruner.masks
pruner_retrain = SparsityPruner(net_retrain)
pruner_retrain.masks = _masks

optimizer = optim.Adam(net_retrain.parameters(), lr=1e-3, weight_decay=5e-4)
train_losses, val_losses, train_accs, val_accs = [], [], [], []

pruner_retrain.apply_mask()
print(net_retrain.param_count())

for epoch in range(N_EPOCH):
    print('Starting epoch {}'.format(epoch+1))
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    train_losses, val_losses, train_accs, val_accs = net_retrain.train_epoch(epoch, optimizer, plot=False, data=plt_data, pruner=pruner_retrain, early_stop=val_acc, LOG=10)

torch.save(net_retrain.state_dict(), './checkpoints/iterative-pruning/{}-reinit-trained'.format(EXPERIMENT_NAME))
save_data = {'train_losses': train_losses, 
             'val_losses': val_losses, 
             'train_accs': train_accs, 
             'val_accs': val_accs}
pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning/{}-reinit.csv'.format(EXPERIMENT_NAME), index=None)

# Retrain from random initialization
print('Retraining from random initialization...')
net_random = Network(trainloader, testloader)
net_random = net_random.to(device)

_masks = pruner.masks
pruner_reinit = SparsityPruner(net_random)
pruner_reinit.masks = _masks

optimizer = optim.Adam(net_random.parameters(), lr=1e-3, weight_decay=5e-4)
train_losses, val_losses, train_accs, val_accs = [], [], [], []

pruner_reinit.apply_mask()
print(net_random.param_count())

for epoch in range(N_EPOCH):
    print('Starting epoch {}'.format(epoch+1))
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    train_losses, val_losses, train_accs, val_accs = net_random.train_epoch(epoch, optimizer, plot=False, data=plt_data, pruner=pruner_reinit, early_stop=val_acc, LOG=10)

torch.save(net_random.state_dict(), './checkpoints/iterative-pruning/{}-rand-init-trained'.format(EXPERIMENT_NAME))
save_data = {'train_losses': train_losses, 
             'val_losses': val_losses, 
             'train_accs': train_accs, 
             'val_accs': val_accs}
pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning/{}-rand-init.csv'.format(EXPERIMENT_NAME), index=None)

print('Done')