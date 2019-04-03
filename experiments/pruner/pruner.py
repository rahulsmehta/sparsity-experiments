import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

class Pruner():
    
    def __init__(self, model, mask_classifier=True):
        super(Pruner, self).__init__()
        self.model = model
        self.masks = {}
        self.mask_classifier = mask_classifier
    
    
    def prune(self, to_retain):
        return
            
    def apply_mask(self):
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                continue
            if self.mask_classifier is not True and 'classifier' in name:
                continue
            if 'linear' in name:
                continue
            mask = self.masks[name]
            param.requires_grad_(requires_grad=False)
            param.mul_(mask)
            param.requires_grad_(requires_grad=True)
            