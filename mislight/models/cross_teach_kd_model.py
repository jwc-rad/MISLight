import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .segmentation_model import SegmentationModel
from .ssl_model import SSLModel
from .losses import define_loss, check_loss
from .losses.utils import softmax_helper, get_current_rampup_weight
from .networks import define_S

class CrossTeachKDModel(SSLModel):
    '''Semi-supervised Learning - Knowledge Distillation from Two Cross Teaching (Cross Pseudo-supervision + Consistency) Teachers
    '''
    @staticmethod
    def get_model_specific_args():
        dummy = argparse.ArgumentParser()
        dummy = CrossTeachKDModel.add_model_specific_args(dummy)
        dummyopt = dummy.parse_args(args=[])
        return vars(dummyopt).keys()
    
    @staticmethod
    def add_model_specific_args(parser):
        parser = SegmentationModel.add_model_specific_args(parser)
        #parser.set_defaults(nsf=[8,16,16], n_stages=[4,5,5])
        
        # segmentation
        parser.add_argument('--segmentation_loss', type=str, default='label_dc_focal', help='loss for segmentation')
        
        # consistency
        parser.add_argument('--consistency_loss', type=str, default='hard_dice', help='loss for consistency')
        parser.add_argument('--consistency_lambda', type=float, default=0.1, help='loss is actually lambda * Gaussian ramp-up function')
        parser.add_argument('--consistency_temperature', type=float, nargs='+', default=[1, 1], help='softmax with temperature scaling')
        parser.add_argument('--consistency_tmax', type=float, default=1, help='tmax for rampup=n_epochs * this')
        parser.add_argument('--consistency_skip_pseudosupervision', action='store_true', help='skip teacher->student pseudosupervision')
        
        # distillation
        parser.add_argument('--distillation_loss', type=str, default='kldiv', help='loss for distillation')
        parser.add_argument('--distillation_lambda', type=float, default=10, help='loss is actually lambda * Gaussian ramp-up function')
        parser.add_argument('--distillation_temperature', type=float, nargs='+', default=[1, 1], help='softmax with temperature scaling')
        parser.add_argument('--distillation_tmax', type=float, default=1, help='tmax for rampup=n_epochs * this')
        
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters() # can access opt by self.hparams['opt']
        
        self.inference_model = opt.inference_model
        if not self.inference:
            self.model_names = ['S', 'T1', 'T2']
        else:
            if self.inference_model == 0.5:
                self.model_names = ['T1', 'T2']
            elif self.inference_model == 0:
                self.model_names = ['T1']
            elif self.inference_model == 1:
                self.model_names = ['T2']
            elif self.inference_model > 1:
                self.model_names = ['S']
        num_nets = 3
        
        self.current_test_idx = None
        self.result_names = ['fpred']
        
        # define network parameters. repeat if 1 value is given
        netargs = SegmentationModel.parse_network_specific_args(opt, num_nets)    
        
        if 'S' in self.model_names:
            self.netS = define_S(opt.nc_input, opt.num_classes, **netargs[0], opt=opt)
        if 'T1' in self.model_names:
            self.netT1 = define_S(opt.nc_input, opt.num_classes, **netargs[1], opt=opt)
        if 'T2' in self.model_names:
            self.netT2 = define_S(opt.nc_input, opt.num_classes, **netargs[2], opt=opt)

        # define training
        if not self.inference:            
            
            # loss
            self.criterionSeg = define_loss(opt.segmentation_loss)
            
            self.criterionCon = define_loss(opt.consistency_loss)
            self.consistency_temperature_x = opt.consistency_temperature[0] if len(opt.consistency_temperature) > 1 else 1
            self.consistency_temperature_y = opt.consistency_temperature[-1]
            self.consistency_lambda        = opt.consistency_lambda
            self.consistency_tmax          = opt.consistency_tmax * opt.n_epochs
            self.consistency_comp = check_loss(opt.consistency_loss, [self.consistency_temperature_x, self.consistency_temperature_y])
            self.consistency_pseudosupervision = not opt.consistency_skip_pseudosupervision
            
            self.criterionDis = define_loss(opt.distillation_loss)
            self.distillation_temperature_x = opt.distillation_temperature[0] if len(opt.distillation_temperature) > 1 else 1
            self.distillation_temperature_y = opt.distillation_temperature[-1]
            self.distillation_lambda        = opt.distillation_lambda
            self.distillation_tmax          = opt.distillation_tmax * opt.n_epochs
            self.distillation_comp = check_loss(opt.distillation_loss, [self.distillation_temperature_x, self.distillation_temperature_y])
            
    
    def _step_S_labeled(self, stage):
        loss = 0
        bs = torch.sum(self.labeled)
        if bs > 0:
            lossS = self.criterionSeg(self.predS[self.labeled], self.label[self.labeled])
            lossT1 = self.criterionSeg(self.predT1[self.labeled], self.label[self.labeled])
            lossT2 = self.criterionSeg(self.predT2[self.labeled], self.label[self.labeled])
            loss_seg = lossS + lossT1 + lossT2
            
            self.log(f'loss/{stage}_lossT1_seg', lossT1, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_lossT2_seg', lossT2, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_lossS_seg', lossS, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_loss_seg', loss_seg, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            loss += loss_seg
            
        # distillation: Ts->S
        w0 = self.distillation_lambda
        if bs > 0 and w0 > 0:
            temp_x = self.distillation_temperature_x
            temp_y = self.distillation_temperature_y
            comp = self.distillation_comp
            tmax = self.distillation_tmax
            rampup = get_current_rampup_weight(self.current_epoch, tmax)
            w = w0 * rampup
            
            loss_dis = self.criterionDis(self.predS[self.labeled]/temp_x, [self.predT1[self.labeled].detach()/temp_y, self.predT2[self.labeled].detach()/temp_y]) * comp
            
            self.log(f'weight/{stage}_weight_dis_labeled', w, batch_size=bs, on_step=False, on_epoch=True)
            self.log(f'loss/{stage}_loss_dis_labeled', loss_dis, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            loss += loss_dis * w              
            
        return loss
        
    def _step_S_unlabeled(self, stage):
        loss = 0
        bs = torch.sum(~self.labeled)
        
        # consistency T1<->T2 + pseudosupervision Ts->S
        w0 = self.consistency_lambda
        if bs > 0 and w0 > 0:
            temp_x = self.consistency_temperature_x
            temp_y = self.consistency_temperature_y
            comp = self.consistency_comp
            tmax = self.consistency_tmax
            rampup = get_current_rampup_weight(self.current_epoch, tmax)
            w = w0 * rampup
            
            lossT1 = self.criterionCon(self.predT1[~self.labeled]/temp_x, self.predT2[~self.labeled].detach()/temp_y) * comp
            lossT2 = self.criterionCon(self.predT2[~self.labeled]/temp_x, self.predT1[~self.labeled].detach()/temp_y) * comp
            if self.consistency_pseudosupervision:
                lossS = self.criterionCon(self.predS[~self.labeled]/temp_x, [self.predT1[~self.labeled].detach()/temp_y, self.predT2[~self.labeled].detach()/temp_y]) * comp
            else:
                lossS = 0
            
            loss_con = lossT1 + lossT2 + lossS

            self.log(f'weight/{stage}_weight_con', w, batch_size=bs, on_step=False, on_epoch=True)
            self.log(f'loss/{stage}_lossT1_con', lossT1, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_lossT2_con', lossT2, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_lossS_con', lossS, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            self.log(f'loss/{stage}_loss_con', loss_con, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            loss += loss_con * w  
        
        # distillation: Ts->S
        w0 = self.distillation_lambda
        if bs > 0 and w0 > 0:
            temp_x = self.distillation_temperature_x
            temp_y = self.distillation_temperature_y
            comp = self.distillation_comp
            tmax1 = self.consistency_tmax
            rampup1 = get_current_rampup_weight(self.current_epoch, tmax1)            
            tmax2 = self.distillation_tmax
            rampup2 = get_current_rampup_weight(self.current_epoch, tmax2)
            w = w0 * rampup1 * rampup2
            
            loss_dis = self.criterionDis(self.predS[~self.labeled]/temp_x, [self.predT1[~self.labeled].detach()/temp_y, self.predT2[~self.labeled].detach()/temp_y]) * comp
            
            self.log(f'weight/{stage}_weight_dis_unlabeled', w, batch_size=bs, on_step=False, on_epoch=True)
            self.log(f'loss/{stage}_loss_dis_unlabeled', loss_dis, batch_size=bs, on_step=(stage!='valid'), on_epoch=True)
            loss += loss_dis * w  

        return loss
