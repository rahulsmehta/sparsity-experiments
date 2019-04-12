import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from .pruner import Pruner

class BERTSparsityPruner(Pruner):
    def prune(self, to_retain, prune_global=True):
        pass

    def apply_mask(self)
        pass