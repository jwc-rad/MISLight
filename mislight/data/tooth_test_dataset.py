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


class ToothTestDataset(ToothBaseDataset):    
    '''Only for Test. For fine-coarse.
    '''
    ## override this to define self.transform
    def prepare_transforms(self):
        self.transform = None
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        if len(self.mask_paths)>0:
            tfm = [
                LoadImaged(keys=['image', 'mask']),
                EnsureChannelFirstd(keys=['image', 'mask']),
                PickChanneld(keys=['image'], channels=3),
                CropForegroundV2d(keys=['image', 'mask'], source_key='mask', margin=0.05),
                ResizeV2d(keys=['image', 'mask'], spatial_size=[self.opt.load_size,]*2 if self.opt.load_mode=='all' else self.opt.load_size, size_mode=self.opt.load_mode, mode=['bilinear', 'nearest-exact']),
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