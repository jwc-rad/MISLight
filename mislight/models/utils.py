import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

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
            return 1.0 - epoch / float(1 + opt.max_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epochs, eta_min=0)
    elif opt.lr_policy == 'none':
        def lambda_rule(epoch):
            return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        def poly_lr(epoch, exponent=0.9):
            return (1 - epoch / opt.max_epochs)**exponent
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_optimizer(net_params, opt):
    if opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net_params, opt.lr, betas=(0.9, 0.999), eps=1e-04)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(net_params, opt.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-04)
    elif opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net_params, opt.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        return NotImplementedError(f'optimizer {opt.optimizer} is not implemented')
    return optimizer

