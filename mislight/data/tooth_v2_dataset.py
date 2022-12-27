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

from monai.transforms import (
    Compose,
    Transform,
    MapTransform,
    AsDiscreted,
    EnsureChannelFirstd,
    GridPatchd,
    Flip,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandGridPatchd,
    Randomizable,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from .transforms import ConvertLabeld, RandHorizontalFlipLabeld, ResizeV2d, RandResizeV2d, RandSpatialPadCropd, PickChanneld, CropForegroundV2d

from .tooth_base_dataset import ToothBaseDataset

class ToothV2Dataset(ToothBaseDataset):    
    ## override this to define self.transform
    def prepare_transforms(self):
        self.transform = None
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        hflip_convert_dict = {
            11: 21,
            12: 22,
            13: 23,
            14: 24,
            15: 25,
            16: 26,
            21: 11,
            22: 12,
            23: 13,
            24: 14,
            25: 15,
            26: 16,
            31: 41,
            32: 42,
            33: 43,
            34: 44,
            35: 45,
            36: 46,
            41: 31,
            42: 32,
            43: 33,
            44: 34,
            45: 35,
            46: 36,
        }
        convert_dict= {
            11: 1,
            12: 2,
            13: 3,
            14: 4,
            15: 5,
            16: 6,
            21: 7,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            31: 13,
            32: 14,
            33: 15,
            34: 16,
            35: 17,
            36: 18,
            41: 19,
            42: 20,
            43: 21,
            44: 22,
            45: 23,
            46: 24,
        }
        
        if self.phase == 'train':
            tfm = [
                LoadImaged(keys=['image', 'mask']),
                EnsureChannelFirstd(keys=['image', 'mask']),
                PickChanneld(keys=['image'], channels=3),
                
                CropForegroundV2d(keys=['image', 'mask'], source_key='mask', margin=0.05),
                RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.5,2)),
                
                RandResizeV2d(keys=['image', 'mask'], spatial_size=[self.opt.load_size,]*2 if self.opt.load_mode=='all' else self.opt.load_size, max_spatial_size=[self.opt.load_max_size,]*2 if self.opt.load_mode=='all' else self.opt.load_max_size, size_mode=self.opt.load_mode, mode=['bilinear', 'nearest-exact']),
                RandSpatialPadCropd(keys=['image', 'mask'], roi_size=self.opt.patch_size, pad_tolerance=self.opt.patch_pad_tolerance),
                
                RandHorizontalFlipLabeld(keys=['image', 'mask'], prob=0.5, convert_dict=hflip_convert_dict),
                ConvertLabeld(keys=['mask'], convert_dict=convert_dict),
                
                NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=img_norm_cfg['mean'], divisor=img_norm_cfg['std']),
                ToTensord(keys=['image', 'mask']),
            ]           
        elif self.phase == 'valid':
            tfm = [
                LoadImaged(keys=['image', 'mask']),
                EnsureChannelFirstd(keys=['image', 'mask']),
                PickChanneld(keys=['image'], channels=3),
                
                CropForegroundV2d(keys=['image', 'mask'], source_key='mask', margin=0.05),
                ResizeV2d(keys=['image', 'mask'], spatial_size=[self.opt.patch_size,]*2 if self.opt.load_mode=='all' else self.opt.patch_size, size_mode=self.opt.load_mode, mode=['bilinear', 'nearest-exact']),
                
                ConvertLabeld(keys=['mask'], convert_dict=convert_dict),
                NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=img_norm_cfg['mean'], divisor=img_norm_cfg['std']),
                ToTensord(keys=['image', 'mask']),
            ]              
        else:
            tfm = [
                LoadImaged(keys=['image']),
                EnsureChannelFirstd(keys=['image']),
                PickChanneld(keys=['image'], channels=3),
                ResizeV2d(keys=['image'], spatial_size=[self.opt.load_size,]*2 if self.opt.load_mode=='all' else self.opt.load_size, size_mode=self.opt.load_mode, mode=['bilinear']),
                NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=img_norm_cfg['mean'], divisor=img_norm_cfg['std']),
                ToTensord(keys=['image']),
            ]    
    
        self.transform = Compose(tfm)