from collections import OrderedDict
import collections.abc
import json
import os
import numpy as np
import pickle
from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms
import pytorch_lightning as pl

from mislight.utils.find_class import recursive_find_python_class

DATA_MODULE = 'mislight.data'

class MyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_class = recursive_find_python_class(self.opt.dataset_mode, DATA_MODULE)
            self.ds_train = dataset_class(self.opt, phase='train')
            print(f"train dataset [{type(self.ds_train).__name__}] was created")  
            self.ds_valid = dataset_class(self.opt, phase='valid')
            print(f"valid dataset [{type(self.ds_valid).__name__}] was created")  

        if stage == "test" or stage is None:            
            dataset_class = recursive_find_python_class(self.opt.dataset_mode, DATA_MODULE)
            self.ds_test = dataset_class(self.opt, phase='test')
            print(f"test datasets [{type(self.ds_test).__name__}] were created")            
            
        # only for valid predict
        if stage == "valid":
            dataset_class = recursive_find_python_class(self.opt.dataset_mode, DATA_MODULE)
            self.ds_valid = dataset_class(self.opt, phase='valid')
            print(f"valid dataset [{type(self.ds_valid).__name__}] was created")              
            
    def train_dataloader(self):
        DL = DataLoader(
            self.ds_train, 
            batch_size=self.opt.batch_size,
            #drop_last=self.opt.batch_drop_last,
            shuffle=True,
            num_workers=int(self.opt.num_workers),
            pin_memory=True,
        )
        return DL
    
    def val_dataloader(self):
        if len(self.ds_valid) > 0:
            DL = DataLoader(
                self.ds_valid, 
                batch_size=self.opt.batch_size_inference,
                #drop_last=self.opt.batch_drop_last,
                shuffle=False,
                num_workers=int(self.opt.num_workers),
                persistent_workers=False,
                pin_memory=False,
            )
            return DL
        else:
            return None
        
    def test_dataloader(self):
        if len(self.ds_test) > 0:
            DL = DataLoader(
                self.ds_test, 
                batch_size=self.opt.batch_size_inference,
                shuffle=False,
                num_workers=int(self.opt.num_workers),
                persistent_workers=False,
                pin_memory=False,
            )
            return DL
        else:
            return None