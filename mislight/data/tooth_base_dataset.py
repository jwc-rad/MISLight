import glob
import importlib
import json
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataset

class ToothBaseDataset(BaseDataset):    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        basedir = self.opt.datadir
        
        self.image_dir = os.path.join(basedir, 'images')
        if 'maskdir' in self.opt and self.opt.maskdir:
            self.mask_dir = self.opt.maskdir
        else:
            self.mask_dir = os.path.join(basedir, 'mask')       
        print(f'preparing {self.image_dir} and {self.mask_dir}')
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        
        _size = min(len(self.image_paths), self.opt.max_dataset_size)
        self.image_paths = self.image_paths[:_size]
        self.mask_paths = self.mask_paths[:_size]
        
        self.keys = [os.path.basename(x).split('.png')[0] for x in self.image_paths]
        
        if self.phase in ['train', 'valid']:
            cfold = int(self.opt.fold[0]) if (isinstance(self.opt.fold, list)) else self.opt.fold
            with open(self.opt.dataset_split, 'rb') as f:
                self.ds_split = pickle.load(f)        
            if self.phase == 'train':
                self.image_paths = [self.image_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['train']]
                self.mask_paths = [self.mask_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['train']]
                self.keys = [x for x in self.keys if x in self.ds_split[cfold]['train']]
            elif self.phase == 'valid':
                self.image_paths = [self.image_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['valid']]
                self.mask_paths = [self.mask_paths[i] for i,x in enumerate(self.keys) if x in self.ds_split[cfold]['valid']]
                self.keys = [x for x in self.keys if x in self.ds_split[cfold]['valid']]
                         
        print(f'image: {len(self.image_paths)}\nmask: {len(self.mask_paths)}')
                    
        print(f'keys: {len(self.keys)}')
    
    ## override this to read data by index. must return image, mask, meta or image, None, meta.
    def read_data(self, index):
        image_path = self.image_paths[index]
        key = self.keys[index]
        im = Image.open(image_path)
        metadata = {'key': key, 'image_path': image_path, 'original_size': np.array(im.size[::-1])}
        
        if len(self.mask_paths)>0:
            mask_path = self.mask_paths[index]            
            metadata['mask_path'] = mask_path
        else:
            mask_path = None
        
        read_items = {
            'image': image_path,
            'metadata': metadata,
        }            
        if mask_path:
            read_items['mask'] = mask_path
        
        return read_items
