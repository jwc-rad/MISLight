import glob
import json
import os
import numpy as np
import random

import torch
import torchvision.transforms

from mislight.utils.misc import find_files
from .base_dataset import BaseDataset, TwoStreamMiniBatchSampler
from .transforms import ToTensor3D

class SegmentationSSLDataset(BaseDataset):
    '''
    '''
    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument('--sample_labeled', type=int, default=1, help='ratio of labeled samples for batch sampler')
        parser.add_argument('--sample_unlabeled',type=int, default=1, help='ratio of unlabeled samples for batch sampler')
        return parser
    
    def __init__(self, opt, keys=None, transforms=None):
        super().__init__(opt)
            
        with open(opt.dataset_json, 'r') as f:
            self.ds = json.load(f)
        if keys is None:
            image_keys = self.ds['image_keys']
        else:
            image_keys = keys
        self.label_keys = [k for k in self.ds['label_keys'] if k in image_keys]
        self.image_keys = self.label_keys + [k for k in image_keys if not (k in self.label_keys)]
        self.dummylabel = self.ds['num_classes'] + 1
        
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
            
        self.batch_size = opt.batch_size
        self.sample_size = (opt.sample_labeled + opt.sample_unlabeled) * opt.batch_size
        self.sample_size_unlabeled = opt.sample_unlabeled * opt.batch_size
                
    def batch_sampler(self, shuffle=True):
        if self.X_size - len(self.label_keys) > 0:
            labeled_idxs = list(range(0, len(self.label_keys)))
            unlabeled_idxs = list(range(len(self.label_keys), self.X_size))
            return TwoStreamMiniBatchSampler(labeled_idxs, unlabeled_idxs, self.sample_size, self.sample_size_unlabeled, self.batch_size, shuffle)
        else:
            return None        
                
    def __len__(self):
        return self.X_size

    def __getitem__(self, index):
        X_path = self.X_paths[index]
        d = self.loader(X_path)
        
        X_img = d['image']
        # set dummy labels to unlabeled cases
        if 'label' in d.keys():
            Y_img = d['label'].astype('uint8')
            labeled = True
        else:
            Y_img = np.full(X_img.shape[-3:], self.dummylabel, dtype='uint8')
            labeled = False
            
        slicer = ()
        for _ in range(3):
            slicer += (slice(random.randint(0,self.coarse_factor - 1), None, self.coarse_factor),)
        X_img = X_img[(slice(None),)*max(0,len(X_img.shape)-3)+slicer]
        Y_img = Y_img[(slice(None),)*max(0,len(Y_img.shape)-3)+slicer]
            
        return_items = self.transform({'image': X_img, 'label': Y_img})
        return_items['labeled'] = labeled
        return return_items
