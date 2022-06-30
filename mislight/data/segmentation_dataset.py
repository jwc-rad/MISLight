import glob
import json
import os
import numpy as np
import random

import torch
import torchvision.transforms

from mislight.utils.misc import find_files
from .base_dataset import BaseDataset
from .transforms import ToTensor3D

class SegmentationDataset(BaseDataset):
    '''Fully Supervised segmentation dataset
    '''
    def __init__(self, opt, keys=None, transforms=None):
        super().__init__(opt)
            
        with open(opt.dataset_json, 'r') as f:
            self.ds = json.load(f)
        if keys is None:
            image_keys = self.ds['image_keys']
        else:
            image_keys = keys

        self.label_keys = [k for k in self.ds['label_keys'] if k in image_keys]
        self.image_keys = self.label_keys
        
        self.X_paths = [os.path.join(self.datadir, f'{x}.{opt.file_extension}') for x in self.image_keys]
        self.X_size = min(len(self.X_paths), opt.max_dataset_size)
        self.X_paths = self.X_paths[:self.X_size]
        self.image_keys = self.image_keys[:self.X_size]
        self.label_keys = self.label_keys[:self.X_size]

        self.coarse_factor = opt.coarse_factor
        
        if transforms is None:
            self.transform = torchvision.transforms.Compose([ToTensor3D()])
        else:
            self.transform = transforms        
        
    def __len__(self):
        return self.X_size

    def __getitem__(self, index):
        X_path = self.X_paths[index]
        d = self.loader(X_path)
        
        slicer = ()
        for _ in range(3):
            slicer += (slice(random.randint(0,self.coarse_factor - 1), None, self.coarse_factor),)
        X_img = d['image']
        Y_img = d['label'].astype('uint8')
        X_img = X_img[(slice(None),)*max(0,len(X_img.shape)-3)+slicer]
        Y_img = Y_img[(slice(None),)*max(0,len(Y_img.shape)-3)+slicer]
            
        return_items = self.transform({'image': X_img, 'label': Y_img})
        return_items['labeled'] = True
        return 
