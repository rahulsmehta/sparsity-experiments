#%matplotlib inline
#import matplotlib.pylab as plt
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from .pruner import Pruner

class SparsityPruner(Pruner):
    def prune(self, to_retain, prune_global=False):
        if not prune_global:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    continue
                if 'linear' in name:
                    continue

                active_weights = torch.abs(param[param != 0]).cpu().data.numpy()
                threshold = np.quantile(active_weights, 1-to_retain)
                mask = torch.abs(param) >= threshold
                mask = mask.float()
                self.masks[name] = mask
                
                param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                param.requires_grad_(requires_grad=True)
        else:
            all_weights = np.array([])
            for name, param in self.model.named_parameters():
                if self.matches(name, ['bias', 'linear', 'bn']): # only prune conv + shortcut layers
                    continue
                active_weights = torch.abs(param[param != 0])
                active_weights = active_weights.view(-1).cpu().data.numpy()
                all_weights = np.concatenate((all_weights, active_weights))

            threshold = np.quantile(all_weights, 1-to_retain)
            for name, param in self.model.named_parameters():
                if self.matches(name, ['bias', 'linear', 'bn']): # only prune conv + shortcut layers
                    continue
                mask = (torch.abs(param) >= threshold).float()
                self.masks[name] = mask
                
                param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                param.requires_grad_(requires_grad=True)

    def apply_mask(self, prune_global=False):
        if not prune_global:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    continue
                if 'linear' in name:
                    continue
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                param.requires_grad_(requires_grad=True)
        else:
            for name, param in self.model.named_parameters():
                if self.matches(name, ['bias', 'linear', 'bn']) or name not in self.masks: # only prune conv + shortcut layers
                    continue
                
                mask = self.masks[name]
                param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                param.requires_grad_(requires_grad=True)