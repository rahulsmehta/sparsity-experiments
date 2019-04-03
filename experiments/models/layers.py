import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
    
class MConv2d(nn.Conv2d):
    def __init__(self, mask=None, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.mask = mask

    def forward(self, x):
        if mask is not None:
            if mask.shape != self.weight.data.shape:
                raise Exception('Dimension mismatch: expected {} but got {}'.format(
                    mask.shape, self.weight.data.shape))
            self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    
class MLinear(nn.Linear):
    def __init__(self, mask, *args, **kwargs):
        super(MLinear, self).__init__(*args, **kwargs)
        self.mask = mask
    
    def forward(self, x):
        if mask is not None:
            if mask.shape != self.weight.data.shape:
                raise Exception('Dimension mismatch: expected {} but got {}'.format(
                    mask.shape, self.weight.data.shape))
            self.weight.data *= self.mask
        return super(MLinear, self).forward(x)