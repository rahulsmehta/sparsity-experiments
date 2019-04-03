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
parser.add_argument('--ft', metavar='P', type=int, default=10,
                    help='Number of epochs to fine-tune per pruning iteration')
# parser.add_argument('--to_retain', metavar='R', type=float, default=0.2,
#                     help='Percentage of params to retain')

parser.add_argument('--log', metavar='L', type=int, default=25,
                    help='Log validation accuracy every L iters')
parser.add_argument('--device', metavar='L', type=str, default='cuda:0',
                    help='GPU device to use')
parser.add_argument('--batch_size', metavar='L', type=int, default='128',
                    help='GPU device to use')



args = parser.parse_args()


EXPERIMENT_NAME = args.name
N_EPOCH = args.epoch
pruning_iter = args.prune
LOG = args.log
n_ft_epochs = args.ft

device = args.device

# Load train and test set
TRAIN_BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = 100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./cifar-data', train=True,
                                     download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar-data', train=False,
                                     download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=2)

Network = ResNet18
print('Lottery Ticket Experiment for {} with pruning rate {} on device {}'.format(Network, 100*(0.8**pruning_iter) ,device))


def get_lr(epoch):
    if (epoch+1) >= 150:
        return 1e-4
    elif (epoch+1) >= 100:
        return 5e-3
    return 1e-3

# Train base network
print('Training base network...')
net_base = Network(trainloader, testloader, device=device)
net_base = net_base.to(device)
torch.save(net_base.state_dict(), './checkpoints/iterative-pruning-cifar10/{}-init'.format(EXPERIMENT_NAME))

#lr_schedule=np.concatenate(([15e-5, 2*15e-5, 4*15e-5, 8*15e-5],np.repeat(8*15e-5, N_EPOCH-4)))
train_losses, val_losses, train_accs, val_accs = [], [], [], []


for epoch in range(N_EPOCH):
    print('Starting epoch {}'.format(epoch+1))
    optimizer = optim.SGD(net_base.parameters(), lr=get_lr(epoch), momentum=0.9, weight_decay=1e-4)
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    train_losses, val_losses, train_accs, val_accs, stopped = net_base.train_epoch(epoch,
        optimizer, plot=False, data=plt_data, LOG=LOG)
    if stopped:
        break

torch.save(net_base.state_dict(), './checkpoints/iterative-pruning-cifar10/{}-trained'.format(EXPERIMENT_NAME))
save_data = {'train_losses': train_losses, 
             'val_losses': val_losses, 
             'train_accs': train_accs, 
             'val_accs': val_accs}
pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning-cifar10/{}-init.csv'.format(EXPERIMENT_NAME), index=None)

# Iterative pruning 
to_retain_iter = 0.8

net_ft = Network(trainloader, testloader, device=device)
net_ft = net_ft.to(device)
net_ft.load_state_dict(torch.load('./checkpoints/iterative-pruning-cifar10/{}-trained'.format(EXPERIMENT_NAME)))
val_acc, _ = net_ft.test()
before_count = net_ft.param_count()

print('Before pruning: {}, params: {}'.format(val_acc, before_count))
pruner = SparsityPruner(net_ft)

optimizer = optim.SGD(net_ft.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
train_losses, val_losses, train_accs, val_accs = [], [], [], []



for prune_epoch in range(pruning_iter):
    plt_data = (train_losses, val_losses, train_accs, val_accs)
    pruner.prune(to_retain_iter, prune_global=True)
    after_count = net_ft.param_count()
    print('Starting pruning iteration {}, pct. remaining : {}'.format(prune_epoch+1, after_count/before_count))
    for ft_epoch in range(n_ft_epochs):
        train_losses, val_losses, train_accs, val_accs, stopped = net_ft.train_epoch(ft_epoch, 
            optimizer, plot=False, data=plt_data, LOG=LOG, pruner=pruner, early_stop=val_acc)
        if stopped:
            break
    pruner_path = 'experiment_data/iterative-pruning-cifar10/{}-{}.p'.format(EXPERIMENT_NAME, prune_epoch+1)
    pickle.dump(pruner, open(pruner_path, "wb" ) )

print('After pruning: {}, params: {}'.format(net_ft.test()[0], after_count))


# Retrain from winning ticket initialization
for prune_epoch in range(pruning_iter):
    pruner_path = "experiment_data/iterative-pruning-cifar10/{}-{}.p".format(EXPERIMENT_NAME, prune_epoch+1)
    print('Retraining from winning ticket initialization at pruning level {}...'.format(prune_epoch+1))
    net_retrain = Network(trainloader, testloader, device=device)
    net_retrain = net_retrain.to(device)
    net_retrain.load_state_dict(torch.load('./checkpoints/iterative-pruning-cifar10/{}-init'.format(EXPERIMENT_NAME)))

    pruner = pickle.load(open(pruner_path, 'rb'))
    _masks = pruner.masks
    pruner_retrain = SparsityPruner(net_retrain)
    pruner_retrain.masks = _masks

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    pruner_retrain.apply_mask(prune_global=True)
    print(net_retrain.param_count())

    for epoch in range(N_EPOCH):
        print('Starting epoch {}'.format(epoch+1))
        optimizer = optim.SGD(net_retrain.parameters(), lr=get_lr(epoch), momentum=0.9, weight_decay=1e-4)
        plt_data = (train_losses, val_losses, train_accs, val_accs)
        train_losses, val_losses, train_accs, val_accs, stop = net_retrain.train_epoch(epoch,
            optimizer, plot=False, data=plt_data, pruner=pruner_retrain, LOG=LOG)
        if stop:
            break

    torch.save(net_retrain.state_dict(), './checkpoints/iterative-pruning-cifar10/{}-{}-reinit-trained'.format(EXPERIMENT_NAME, prune_epoch+1))
    save_data = {'train_losses': train_losses, 
                'val_losses': val_losses, 
                'train_accs': train_accs, 
                'val_accs': val_accs}
    pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning-cifar10/{}-{}-reinit.csv'.format(EXPERIMENT_NAME, prune_epoch+1), index=None)

# Retrain from random initialization
# pruner_path = "experiment_data/iterative-pruning-cifar10/{}.p".format(EXPERIMENT_NAME)
# pruner = pickle.load(open(pruner_path, 'rb'))

# print('Retraining from random initialization...')
# net_random = Network(trainloader, testloader, device=device)
# net_random = net_random.to(device)

# _masks = pruner.masks
# pruner_reinit = SparsityPruner(net_random)
# pruner_reinit.masks = _masks

# train_losses, val_losses, train_accs, val_accs = [], [], [], []

# pruner_reinit.apply_mask(prune_global=True)
# print(net_random.param_count())

# for epoch in range(N_EPOCH):
#     print('Starting epoch {}'.format(epoch+1))
#     optimizer = optim.SGD(net_random.parameters(), lr=get_lr(epoch), momentum=0.9, weight_decay=1e-4)
#     plt_data = (train_losses, val_losses, train_accs, val_accs)
#     train_losses, val_losses, train_accs, val_accs, stopped = net_random.train_epoch(epoch, optimizer, plot=False, data=plt_data, pruner=pruner_reinit, LOG=LOG)
#     if stopped:
#         break

# torch.save(net_random.state_dict(), './checkpoints/iterative-pruning-cifar10/{}-rand-init-trained'.format(EXPERIMENT_NAME))
# save_data = {'train_losses': train_losses, 
#              'val_losses': val_losses, 
#              'train_accs': train_accs, 
#              'val_accs': val_accs}
# pd.DataFrame(save_data).to_csv('./experiment_data/iterative-pruning-cifar10/{}-rand-init.csv'.format(EXPERIMENT_NAME), index=None)

print('Done')


