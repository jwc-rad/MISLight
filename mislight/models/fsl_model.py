import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .segmentation_model import SegmentationModel
from .losses import define_loss
from .networks import define_S

class FSLModel(SegmentationModel):
    '''Fully Supervised Learning
    '''
    @staticmethod
    def get_model_specific_args():
        dummy = argparse.ArgumentParser()
        dummy = FSLModel.add_model_specific_args(dummy)
        dummyopt = dummy.parse_args(args=[])
        return vars(dummyopt).keys()
    
    @staticmethod
    def add_model_specific_args(parser):
        parser = SegmentationModel.add_model_specific_args(parser)
                
        # loss
        parser.add_argument('--segmentation_loss', type=str, default='label_dc_focal', help='loss for segmentation')
        
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters() # can access opt by self.hparams['opt']
        
        self.model_names = ['S']
        num_nets = 1
        
        self.current_test_idx = None
        self.result_names = ['fpred']
        
        # define network parameters. repeat if 1 value is given
        netargs = SegmentationModel.parse_network_specific_args(opt, num_nets)
        
        self.netS = define_S(opt.nc_input, opt.num_classes, **netargs[0])
                
        # define training
        if not self.inference:
            # loss
            self.criterionSeg = define_loss(opt.segmentation_loss)
        
    def set_input(self, batch):
        self.image = batch['image']
        self.current_batch_size = batch['image'].shape[0]
        if not self.inference:
            self.label = batch['label']

    def _step_S(self, stage):
        loss = self.criterionSeg(self.predS, self.label)
        self.log(f'loss/{stage}_loss', loss, batch_size=self.current_batch_size, on_step=False, on_epoch=True)
        return loss
