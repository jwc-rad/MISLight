import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, opt, phase='train'):
        super().__init__()
        self.opt = opt
        self.phase = phase
        self.prepare_data()
        self.prepare_transforms()
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        read_items = self.read_data(index)

        return_items = self.transform(read_items)
        
        return return_items
    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        pass
    
    ## override this to read data by index
    def read_data(self, index):
        pass
    
    ## override this to define self.transform
    def prepare_transforms(self):
        pass