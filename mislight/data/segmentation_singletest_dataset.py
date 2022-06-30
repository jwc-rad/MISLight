import glob
import json
import os
import numpy as np
import random

import torch
import torchvision.transforms

from mislight.utils.misc import find_files
from .base_dataset import BaseDataset
from .transforms import Padding3D, RandomCrop3D, ClipHU, Normalize, CreateOnehotLabel, ToTensor3D


class SegmentationSingleTestDataset(BaseDataset):
    '''Dataset for inference of single preprocessed case
    '''
    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument('--slidingwindow_overlap', type=float, default=0.5, help='overlap ratio of sliding window')
        return parser
    
    def __init__(self, opt, key, transforms=None):
        super().__init__(opt)
        self.key = key
        self.X_path = os.path.join(self.datadir, f'{key}.{opt.file_extension}')
        self.coarse_factor = opt.coarse_factor
        
        X_data = self.loader(self.X_path)
        X_img = X_data['image']
                
        # padding for sliding window. sliding window size = coarse_factor * crop size * (1 - overlapping ratio)
        crop_size = np.array(opt.crop_size) * self.coarse_factor
        sliding_window_size = (crop_size*(1-opt.slidingwindow_overlap)).astype(int)
        old_sh = np.array(X_img.shape[-3:])
        new_sh = (np.ceil(np.divide(np.array(X_img.shape[-3:]), sliding_window_size)) * sliding_window_size).astype(int)
        pads = new_sh - old_sh
        pads = np.array([[a//2, a-a//2] for a in pads.astype(int)])
        
        # sliding window indexes
        slices = np.divide(new_sh, sliding_window_size).astype(int) - 1
        slices[slices<0] = 0
        #zs, ys, xs = np.meshgrid(*[np.arange(s) for s in slices])
        slices = np.stack([mg.ravel() for mg in np.meshgrid(*[np.arange(s) for s in slices])], axis=-1) * sliding_window_size
        slices = np.concatenate([slices, slices+crop_size], axis=-1)
                
        self.original_shape = np.array(X_img.shape[-3:])
        self.padding = pads
        self.X_slices = slices
        self.X_size = len(self.X_slices)
        
        if transforms is None:
            self.transform = torchvision.transforms.Compose([ToTensor3D()])
        else:
            self.transform = transforms
        
    def __len__(self):
        return self.X_size

    def __getitem__(self, index):
        X_data = self.loader(self.X_path)        
        X_img = X_data['image']  
        X_img = np.pad(X_img, ((0,0),)*max(0,len(X_img.shape)-3) + tuple([tuple(p) for p in self.padding]), 'constant', constant_values=-1024)
        
        s1, s2, s3, e1, e2, e3 = slice_parameters = self.X_slices[index]
        slice_func = (slice(None),slice(s1, e1), slice(s2, e2), slice(s3, e3))
        
        X_img = X_img[slice_func]
        slicer = ()
        for _ in range(3):
            slicer += (slice(None, None, self.coarse_factor),)
        X_img = X_img[(slice(None),)*max(0,len(X_img.shape)-3)+slicer]

        return_items = self.transform({'image': X_img})
        return_items['slice'] = torch.from_numpy(slice_parameters)
        return_items['padding'] = torch.from_numpy(self.padding)
        return_items['original_shape'] = torch.from_numpy(self.original_shape)
                
        return return_items
