import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

from .norb_loader import *

class NORB(Dataset):
    def __init__(self, root, transform, train=True, *args, **kwargs):
        super(NORB, self).__init__(*args, **kwargs)
        self.root = root
        self.train = train
        self.transform = transform
        self.norb_loader = SmallNORBDataset(dataset_root=self.root)
        self.example_list = self.norb_loader.data['train'] if self.train else self.norb_loader.data['test']
        self.data_len = len(self.example_list)

        # Preprocessing transforms
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize((32, 32))

        
    def __getitem__(self, index):
        ex = self.example_list[index]
        img = ex.image_lt
        img = img.reshape(96, 96, 1)
        img = self.to_pil(img)
        img = img.convert('RGB')
        img = self.resize(img)
        user_transform = self.transform(img)
        label = ex.category.astype(np.long)
        return (user_transform, label)

    def __len__(self):
        return self.data_len
