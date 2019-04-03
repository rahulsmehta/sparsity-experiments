import torch.nn as nn
import matplotlib.pylab as plt
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .plot import plot_training

class BaseModel(nn.Module):
    def __init__(self, trainloader, testloader, device='cuda'):
        super(BaseModel, self).__init__()
        self.device = device
        
        # Train & test sets
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.LOG = 25
        
    def forward(self, x):
        return x
    
    def param_count(self):
        total = 0
        for name,param in self.named_parameters():
            total += torch.norm(param, p=0).item()
        return total
            
    
    """
    Run a training iteration 
    """
    def train_step(
        self,
        epoch,
        batch_idx,
        inputs,
        labels,
        optimizer,
        criterion=None
    ):
        criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        optimizer.zero_grad()
        
        outputs = self(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss, outputs
    
    
    def train_epoch(
        self,
        epoch,
        optimizer,
        plot=False,
        data=None,
        LOG=10,
        pruner=None,
        early_stop=None
    ):
        self.train()
        self.LOG = LOG
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        if data is not None:
            train_losses, val_losses, train_accs, val_accs = data
            
        running_loss = 0.
        for batch_idx, data in enumerate(self.trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            loss, outputs = self.train_step(epoch, batch_idx, inputs, labels, optimizer)
            if pruner is not None:
                pruner.apply_mask(prune_global=True)
            
            
            running_loss += loss.item()
            if (batch_idx+1) % self.LOG == 0:
                train_loss = running_loss/self.LOG
                train_losses.append(train_loss)
                running_loss = 0.
                
                preds = outputs.argmax(dim=1)
                correct = (labels.eq(preds)).sum()
                train_acc = correct.float()/len(labels)
                train_accs.append(train_acc.item())
                
                val_acc, val_loss = self.test()
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if early_stop is not None:
                    if type(val_acc) == float:
                        if val_acc >= early_stop-(5e-3):
                            print('Stopping at iter {} in epoch {}'.format(batch_idx+1, epoch+1))
                            return train_losses, val_losses, train_accs, val_accs, True
                    else:
                        if val_acc.item() >= early_stop-(1e-2):
                            print('Stopping at iter {} in epoch {}'.format(batch_idx+1, epoch+1))
                            return train_losses, val_losses, train_accs, val_accs, True
                    
                if plot is True:
                    fmt = '[epoch:{},batch:{}]: loss(train): {:.3f}, acc(train): {:.3f}, loss(val): {:.3f}, acc(val): {:.3f}'
                    print(fmt.format(epoch+1,batch_idx+1,train_loss,
                                     train_acc,val_loss,val_acc))
                elif plot == 'loss':
                    plot_training(self.LOG, train_losses, val_losses, title='cross-entropy loss', ylabel='loss')
                elif plot == 'acc':
                    plot_training(self.LOG, train_accs, val_accs, title='lottery ticket convergence', ylabel='accuracy')
                    

        return train_losses, val_losses, train_accs, val_accs, False
            
    
    """
    Return average validation loss and validation accuracy
    """
    def test(self, criterion=None):
        criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.eval()
        tcorrect = 0.
        running_loss = 0.
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.testloader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                # print(outputs.device)
                predictions = outputs.argmax(dim=1)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                bcorrect = labels.eq(predictions).sum()
                tcorrect += bcorrect.item()
        acc = tcorrect/(len(self.testloader)*self.testloader.batch_size)
        avg_loss = running_loss/len(self.testloader)
        return acc, avg_loss
                