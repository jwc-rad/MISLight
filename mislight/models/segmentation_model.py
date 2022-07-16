import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from mislight.utils.image_resample import order2mode
from .base_model import BaseModel
from .losses.utils import softmax_helper

###############################################################################
# Helper Function from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# Poly LR scheduler from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/learning_rate/poly_lr.py
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == 'none':
        def lambda_rule(epoch):
            return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        def poly_lr(epoch, exponent=0.9):
            return (1 - epoch / opt.n_epochs)**exponent
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class SegmentationModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.default_return_layers = False
        
        # inference opts
        self.no_gaussian_weight = opt.no_gaussian_weight
        self.coarse_factor = opt.coarse_factor
    
    @staticmethod
    def add_network_specific_args(parser):
        # network
        parser.add_argument('--netS', type=str, default=['unet'], nargs='+', help='')
        parser.add_argument('--nsf', type=int, default=[16], nargs='+', help='base_num_features')
        parser.add_argument('--n_stages', type=int, default=[5], nargs='+', help='# of stages (pooling) in UNet')
        parser.add_argument('--n_blocks_per_stage', type=int, default=[2], nargs='+', help='# of blocks per stage in UNet')
        parser.add_argument('--max_features', type=int, default=[512], nargs='+', help='max_features in UNet')
        parser.add_argument('--transposed_conv', type=int, default=[1], nargs='+', help='upsampling method. 1 for transposed convolution, 0 for resize-convolution')
        parser.add_argument('--nonlin', type=str, default=['LeakyReLU'], nargs='+', help='nonlinearity [LeakyReLU | ReLU]')
        parser.add_argument('--norm', type=str, default=['in'], nargs='+', help='instance normalization or batch normalization [in | bn]')
        parser.add_argument('--init_type', type=str, default=['xavier'], nargs='+', help='network initialization [xavier | kaiming]')
        
        return parser

    @staticmethod
    def add_model_specific_args(parser):
        # dataset
        parser.add_argument('--batch_drop_last', action='store_true', help='drop_last for dataloader')
        parser.add_argument('--onehotlabel', action='store_true', help="don't touch. default is no onehotlabel")
        parser.add_argument('--coarse_factor', type=int, default=1, help='do not touch. not used')
        parser.add_argument('--crop_size', type=int, default=[96,64,96], nargs=3, help='crop size zyx dimension')
        parser.add_argument('--windowHU', type=int, nargs=2, help='window width, level (HU)')
        parser.add_argument('--no_normalize', action='store_true')
        parser.add_argument('--global_mean', type=float, help='global X for normalize (after windowing)')
        parser.add_argument('--global_std', type=float, help='global X for normalize (after windowing)')
        
        # network
        parser = SegmentationModel.add_network_specific_args(parser)

        return parser
    
    @staticmethod
    def parse_network_specific_args(opt, n_repeat=1):
        dummy = argparse.ArgumentParser()
        dummy = SegmentationModel.add_network_specific_args(dummy)
        dummyopt = dummy.parse_args(args=[])
        
        # define network parameters. repeat if 1 value is given
        npt = {}
        for k in vars(dummyopt).keys():
            v = opt.__dict__[k]
            if (not hasattr(v, '__iter__')) or isinstance(v, str):
                v = [v]
            else:
                v = list(v)
            npt[k] = (v * n_repeat)[:n_repeat]
        
        # additional
        npt['transposed_conv'] = [x>0 for x in npt['transposed_conv']]
        
        parsed_args = []
        for i in range(n_repeat):
            parsed_args.append({k: v[i] for k,v in npt.items()})
        
        return parsed_args   
    
    ### shared basic methods
    
    def configure_optimizers(self):
        netparams = [getattr(self, f'net{m}').parameters() for m in self.model_names]
        optimizer_S = torch.optim.SGD(itertools.chain(*netparams), self.hparams['opt'].lr, weight_decay=self.hparams['opt'].weight_decay, momentum=self.hparams['opt'].momentum, nesterov=(not self.hparams['opt'].no_nesterov))
        
        optimizers = [optimizer_S]
        schedulers = [get_scheduler(optimizer, self.hparams['opt']) for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def forward(self, x, return_layers=False):
        outputs = tuple()
        for m in self.model_names:
            outputs += (getattr(self, f'net{m}')(x, return_layers),)
            
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
        
    def forward_test(self, x):
        preds = self.forward(x)
        if isinstance(preds, (list, tuple)):
            pred_test = softmax_helper(preds[0].detach())
            for p in preds[1:]:
                pred_test += softmax_helper(p.detach())
            pred_test = pred_test / len(preds)
        else:
            pred_test = softmax_helper(preds.detach())
        return pred_test
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        if not self.inference:
            preds = self.forward(self.image, self.default_return_layers)
            if len(self.model_names) == 1:
                m = self.model_names[0]
                if self.default_return_layers:
                    setattr(self, f'pred{m}', preds[0])
                    setattr(self, f'feat{m}', preds[1])
                else:
                    setattr(self, f'pred{m}', preds)
            else:
                for m, p in zip(self.model_names, preds):
                    if self.default_return_layers:
                        setattr(self, f'pred{m}', p[0])
                        setattr(self, f'feat{m}', p[1])
                    else:
                        setattr(self, f'pred{m}', p)
        else:
            self.pred_test = self.forward_test(self.image) 
    
    def training_step(self, batch, batch_idx):
        self._step_forward(batch, batch_idx)    
        result = None
        
        # optimizer_S
        stage = 'train'
        result = self._step_S(stage)
        
        return result
    
    def validation_step(self, batch, batch_idx):
        self._step_forward(batch, batch_idx)
        result = None
        
        # optimizer_S
        stage = 'valid'
        result = self._step_S(stage)
                    
        return result
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):           
        self._step_forward(batch, batch_idx)
        float_dtype = np.float16
        if self.coarse_factor > 1:
            mode = order2mode(self.hparams['opt'].ipl_order_mask)
            #mode = 'nearest'
            sh = torch.tensor(self.pred_test.shape)
            newsize = sh[2:]*self.coarse_factor
            pred_act = F.interpolate(self.pred_test, size=newsize, mode=mode, align_corners=True)
            #pred_act = F.interpolate(self.pred_test, scale_factor=(self.coarse_factor,)*3, mode=mode, align_corners=True)
            pred_act = pred_act.cpu().numpy().astype(float_dtype)
        else:
            pred_act = self.pred_test.cpu().numpy().astype(float_dtype)
        
        # on change of dataloader_idx, calculate final prediction and reset predictions
        if self.current_test_idx != dataloader_idx:
            if dataloader_idx > 0:
                self._process_final_prediction()
            self.current_test_idx = dataloader_idx        
            self.case = np.zeros([pred_act.shape[1]] + batch['original_shape'][0].detach().cpu().numpy().tolist(), dtype=float_dtype)
            self.count = np.zeros(batch['original_shape'][0].detach().cpu().numpy(), dtype=float_dtype)
        
        for i, slc in enumerate(batch['slice'].detach().cpu().numpy()):
            pad = batch['padding'][i].detach().cpu().numpy()
            pad = np.tile(pad[:,0], 2)
            
            slc_case = slc - pad
            slc_case[slc_case<0] = 0
            s1, s2, s3, e1, e2, e3 = slc_case
            slc_case_func = (slice(None), slice(s1, e1), slice(s2, e2), slice(s3, e3))
            
            slc_pred_origin = (slc_case - slc + pad)[-3:]
            slc_pred_crop = np.array(self.case[slc_case_func].shape[-3:])
            
            slc_pred = list(slc_pred_origin) + list(slc_pred_origin + slc_pred_crop)
            s1, s2, s3, e1, e2, e3 = slc_pred
            slc_pred_func = (slice(None), slice(s1, e1), slice(s2, e2), slice(s3, e3))      


            cpred = pred_act[i][slc_pred_func]
            if self.no_gaussian_weight:
                weight = np.array([1])
            else:
                weight = self._get_gaussian(cpred.shape[-3:], sigma_scale=1. / 8, float_dtype=float_dtype)
                weight = np.stack([weight]*cpred.shape[0], axis=0)
            self.case[slc_case_func] += cpred * weight
            self.count[slc_case_func[1:]] += weight[0]
        
    # calculate final prediction for the last case
    def test_epoch_end(self, outputs):
        self._process_final_prediction()
        
        
    def _process_final_prediction(self):
        self.count[self.count==0] = 1
        self.fpred = np.divide(self.case, np.expand_dims(self.count, axis=0))
        # normalize to x/SUM(x, axis=channel_axis)
        channel_sum = np.sum(self.fpred, axis=0, keepdims=True)
        self.fpred = np.divide(self.fpred, channel_sum)
        
    # copied from https://github.com/MIC-DKFZ/nnUNet/nnunet/network_architecture/neural_network.py#L246
    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8, float_dtype=np.float32) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0).astype(float_dtype)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map
