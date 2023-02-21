import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np
from typing import Any, Callable, Dict, List

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

def instantiate_scheduler(optimizer, cfg: DictConfig):
    def add_optimizer(optimizer, cfg):
        if isinstance(cfg, Dict):
            for k, v in cfg.items():
                cfg[k] = add_optimizer(optimizer, v)
            if '_target_' in cfg:
                if issubclass(get_class(cfg.get('_target_')), lr_scheduler._LRScheduler):
                    cfg.update(dict(optimizer=optimizer))
            return cfg
        elif isinstance(cfg, List):
            return [add_optimizer(optimizer, x) for x in cfg]
        else:
            return cfg
    
    _cfg = OmegaConf.to_container(cfg, resolve=True)
    _cfg = add_optimizer(optimizer, _cfg)
    return instantiate(_cfg)


### deprecated
def define_scheduler(optimizer, opt):
    """Return a learning rate scheduler
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

### deprecated
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

