from collections import OrderedDict
import json
import os
import numpy as np
import pickle
from sklearn.model_selection import KFold
from typing import Optional
from torch.utils.data import DataLoader
import torchvision.transforms
import pytorch_lightning as pl

from mislight.utils.misc import find_file
from .base_dataset import find_dataset_using_name
from .transforms import get_transforms

class MyDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parser):
        return parser
    
    def __init__(self, opt):
        super().__init__()
        with open(opt.dataset_json, 'r') as f:
            self.ds = json.load(f)
            
        self.isTrain = opt.isTrain
        if self.isTrain:
            if opt.fold >= 0:
                opt.save_dir = os.path.join(opt.save_dir, f'fold{opt.fold}')
            os.makedirs(opt.save_dir, exist_ok=True)
        
        self.opt = opt
        self.keys_train = None
        self.keys_valid = None
        self.keys_test = None
        self.prepare_data()
        
    def prepare_data(self):
        # keys
        nofold = False
        if 'fold' in self.opt:
            if self.opt.fold >= 0:
                if not os.path.exists(self.opt.dataset_split):
                    self.do_split()
                with open(self.opt.dataset_split, 'rb') as f:
                    splits = pickle.load(f)
                self.keys_train, self.keys_valid = splits[self.opt.fold].values()
                self.keys_train = sorted(self.keys_train)
                self.keys_valid = sorted(self.keys_valid)
                self.keys_test = self.keys_valid
            else:
                nofold = True
        else:
            nofold = True
        if nofold:
            self.keys_valid = []
            self.keys_test = sorted(self.ds['image_keys'])
        
        #### transforms      
        if self.isTrain:
            train_transform = get_transforms('train', self.opt)
            self.train_transform = torchvision.transforms.Compose(train_transform)
            valid_transform = get_transforms('valid', self.opt)
            self.valid_transform = torchvision.transforms.Compose(valid_transform)
        test_transform = get_transforms('test', self.opt)
        self.test_transform = torchvision.transforms.Compose(test_transform)
        
    def do_split(self):
        # 5-fold split for labeled cases
        # nonlabeled cases are always in "train" split
        labeled = sorted(self.ds['label_keys'])
        unlabel = sorted([x for x in self.ds['image_keys'] if not x in labeled])
        
        splits = []
        kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
        for i, (train_idx, valid_idx) in enumerate(kfold.split(labeled)):
            train_keys = np.array(labeled)[train_idx].tolist()
            valid_keys = np.array(labeled)[valid_idx].tolist()
            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys + unlabel
            splits[-1]['valid'] = valid_keys
        
        with open(self.opt.dataset_split, 'wb') as f:
            pickle.dump(splits, f)
        print(f'new split ... {self.opt.dataset_split}')
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_class = find_dataset_using_name(self.opt.dataset_mode)
            self.ds_train = dataset_class(self.opt, keys=self.keys_train, transforms=self.train_transform)
            self.ds_valid = dataset_class(self.opt, keys=self.keys_valid, transforms=self.valid_transform)
            print(f"train dataset [{type(self.ds_train).__name__}] was created")  
            print(f"valid dataset [{type(self.ds_valid).__name__}] was created")  

        if stage == "test" or stage is None:            
            dataset_class = find_dataset_using_name(self.opt.dataset_mode_infer)
            self.ds_test = []
            for k in self.keys_test:
                self.ds_test.append(dataset_class(self.opt, k, transforms=self.test_transform))
            print(f"test datasets [{type(self.ds_test[0]).__name__}] were created")            
            

    def train_dataloader(self):
        batch_sampler = self.ds_train.batch_sampler(True)
        if batch_sampler is None:
            DL = DataLoader(
                self.ds_train, 
                batch_size=self.opt.batch_size,
                drop_last=self.opt.batch_drop_last,
                shuffle=True,
                num_workers=int(self.opt.num_threads),
                pin_memory=True,
            )
        else:
            DL = DataLoader(
                self.ds_train,
                batch_sampler=batch_sampler,
                num_workers=int(self.opt.num_threads),
                pin_memory=True,
            )
        return DL
    
    def val_dataloader(self):
        if len(self.ds_valid) > 0:
            batch_sampler = self.ds_valid.batch_sampler(False)
            if batch_sampler is None:
                DL = DataLoader(
                    self.ds_valid, 
                    batch_size=self.opt.batch_size,
                    drop_last=self.opt.batch_drop_last,
                    shuffle=False,
                    num_workers=int(self.opt.num_threads),
                    pin_memory=True,
                )
            else:
                DL = DataLoader(
                    self.ds_valid,
                    batch_sampler=batch_sampler,
                    num_workers=int(self.opt.num_threads),
                    pin_memory=True,
                )
            return DL
        else:
            return None
        
    def test_dataloader(self):
        if len(self.ds_test) > 0:
            loaders = []
            for ds in self.ds_test:
                batch_sampler = ds.batch_sampler(False)
                if batch_sampler is None:
                    DL = DataLoader(
                        ds, 
                        batch_size=self.opt.batch_size,
                        shuffle=False,
                        num_workers=int(self.opt.num_threads),
                        pin_memory=True,
                    )
                else:
                    DL = DataLoader(
                        ds,
                        batch_sampler=batch_sampler,
                        num_workers=int(self.opt.num_threads),
                        pin_memory=True,
                    )
                loaders.append(DL)
            return loaders
        else:
            return None
