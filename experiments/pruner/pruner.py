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
    def matches(self, name, tokens):
        for t in tokens:
            if t in name:
                return True
        return False
    
    def __init__(self, model, mask_classifier=True):
        super(Pruner, self).__init__()
        self.model = model
        self.masks = {}
        self.mask_classifier = mask_classifier
    
    
    def prune(self, to_retain):
        pass 
            
    def apply_mask(self):
        pass
            